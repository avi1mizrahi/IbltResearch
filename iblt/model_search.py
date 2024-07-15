import math
import random
import subprocess
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path


class Search:
    def __init__(self, entries, goal: Fraction, addressing=None):
        assert entries > 0
        self.entries = entries
        self.goal: Fraction = goal
        # this constant is a stop condition for binary search
        # sometimes params have a mean of the desired value. In that
        # case we'll stop when the mean is with a relatively
        # tight confidence interval
        self.ci_limit = (1-self.goal)/5
        self.addressing = addressing

        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        self.results_dir = results_dir

    def size(self,items, hedge, hashes):
      """ Returns the size of an IBLT (in row) given key params """
      if hedge == 0:
          return 0
      items = math.trunc(items*hedge)
      if self.addressing=='buckets':
        while hashes * (items//hashes) != items:
            items += 1
        return items
      else:
        return items

    def refit(self,items, hedge, hashes):
      """ Returns a hedge value that results in the same    """
      s = self.size(items, hedge, hashes)
      # assert s == self.size(items, s/items, hashes)
      return s/items

    def trial(self, hedge, k):
        """ Determine the decode rate using at least 5000 trials.
        Trials stop when 95% confidence interval is met, or mean is within .99999 to 1.00001  """

        addr = 1 if self.addressing == 'buckets' else 0
        cmd = tuple(map(str, [
            './search',
            self.entries,
            k,
            (hedge * self.entries),
            self.goal.numerator,
            self.goal.denominator,
            addr
        ]))

        returned = subprocess.check_output(cmd)  # returns the exit code in unix
        
        successes,trials=returned.split(b',')
        successes=int(successes)
        trials=int(trials)
        prob=successes/trials
        if successes < trials:
            ci = 1.96*math.sqrt(prob*(1-prob)/trials)
        else:
            ci = -(math.exp(math.log(.05)/trials)-1)

        # print("%d, %d, %.3f" % (int(successes),int(trials),float(prob)))
        # ssize = hedge*self.entries
        return successes, trials, prob, ci

              
    def find_hedge_for_k(self, k):
        """ Binary search for best hedge value given items and number of hashvalues k
        best_size: if passed in, we prune the search when the smallest size is larger 
        """
        assert k >= 2
        high = 20.0 # starting high point binary search (code will double if too low)
        # for some desired decode rates, 20 will be too low for a starting value
        low = 0.0  # min hedge to consider
        epsilon = 0 # difference in number of rows between high and low hedge (in terms of IBLT rows)
        vals = defaultdict(list)
        # Loop until low and high are the same. We move either low up, or high down, each iteration
        # In some cases, we'll move high up. 
        cnt=0
        while self.size(self.entries, high, k) - self.size(self.entries, low, k) > epsilon:
            # Print progress as we go
            cnt+=1
            # print(self.entries,k,self.addressing,cnt,"%3d: %3f(%d)\t%3f\t(%3d)\t%3f(%d)" %
            #       (k, low, self.size(self.entries, low, k),
            #        (high+low)/2, self.size(self.entries, (high+low)/2, k),
            #        high, self.size(self.entries, high, k)))
            
            if cnt>200:
              exit()

            hedge = self.refit(self.entries, (high+low)/2, k)
            # Keep track of which results we've completed.
            if vals[self.size(self.entries, hedge, k)] == list():
                vals[self.size(self.entries, hedge, k)] = self.trial(hedge, k)
            successes, trials, prob, ci= vals[self.size(self.entries, hedge, k)]

            if (prob+ci <= self.goal) or (prob-ci> self.goal-self.ci_limit and 
                                            prob+ci<self.goal+self.ci_limit):
                # failure, try going higher
                old_low = low
                low = self.refit(self.entries, (high+low)/2, k)
                # we've got to increase by at least 1 (i.e.,k rows in the IBLT)
                while self.refit(self.entries, old_low, k) == self.refit(self.entries, low, k):
                    low += 1
            
            # success if we are above the goal, or we are within a tight confidence interval
            elif (prob-ci >= self.goal):
                # success, try going lower
                old_high = high
                high = self.refit(self.entries, (high+low)/2, k)
                # if lower isn't any lower, then we are done
                if self.refit(self.entries, old_high, k) == self.refit(self.entries, high, k):
                    # we are done
                    break
            else:
                print("Huh?","\n1: ",(prob+ci <= self.goal) ,(prob-ci> self.goal-self.ci_limit and
                                            prob+ci<self.goal+self.ci_limit), "\n2: ",prob-ci,self.goal,prob-ci >= self.goal)
                print(prob,ci,self.goal,successes,trials,self.ci_limit,high,low,k,hedge)
                exit()
        # assert for debugging
        assert (prob-ci >= self.goal) or( prob-ci > self.goal-self.ci_limit) or (prob+ci <= self.goal)
        return hedge, successes, trials

    def out_filename_format(self):
        return f'{self.entries}-{float(self.goal)}-{self.addressing}-{{gen}}.csv'

    def run(self, skip_exist=True):
        filename_format = self.out_filename_format()
        if skip_exist and (exists := next(self.results_dir.glob(filename_format.format(gen='*')), None)) is not None:
            print(f'skipping because {exists} was found')
            return

        out_path = self.results_dir / filename_format.format(gen=random.getrandbits(32))

        with out_path.open('w') as fd:
          for k in list(range(3,13)):
              h, successes, trials = self.find_hedge_for_k(k)
              if h is None:
                  continue
              iblt_size = self.size(self.entries, h, k)
              fd.write(
                  f'{self.entries:d}, {float(self.goal):f}, {k:d}, {h:f}, {iblt_size:d}, {successes:d}, {trials:d}\n')
              fd.flush()  # since they runs can take a while

        print('finished', out_path)

#
# def fixed_k_h(addressing):
#   addr = 1 if addressing=='buckets' else 0
#   cmd = ['./search',str(self.entries),str(4),str(1.5*self.entries),sys.argv[3],sys.argv[4],str(addr)]
#   returned = subprocess.check_output(cmd)  # returns the exit code in unix
#   successes,trials=returned.split(b',')
#   items = int(sys.argv[1]) + int(sys.argv[2])
#   # desired decode rate
#   desired_prob = float(sys.argv[3])/float(sys.argv[4])


if __name__ == '__main__':
    items = int(sys.argv[1]) + int(sys.argv[2])
    # desired decode rate
    desired_prob = Fraction(int(sys.argv[3]), int(sys.argv[4]))

    assert items >= 1
    assert desired_prob > 0
    assert desired_prob < 1
    # cProfile.run('Search(items, desired_prob).run()')
    addressing = sys.argv[5] if len(sys.argv) > 5 else None
    Search(items, desired_prob, addressing).run(skip_exist=False)

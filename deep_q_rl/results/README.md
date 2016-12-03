Here are the file names:

(1) breakout_11-25-20-26_.csv and the corresponding files pertain to DQN with
human-boosted data, on the NIPS settings except using 40000 steps per epoch,
which is consistent with the baseline. This is NOT supposed to be finalized,
because it's missing the following:

- The human net was trained on sequences with frame skips 2, 3, and 4, but it
  should only be 4.
- The human net was trained on data which did NOT do the consecutive
  maximizations (to stop flickering effects).
- The human net was trained on data using my grayscale conversion, and not what
  the ALE would have returned. In fact, it's not even important that this
  matches what Google DeepMind did --- all that matters is that the conversion
  is the same as what SPRAGNUR'S code did, because his code is converting it to
  grayscale.

Despite these flaws, this still looks like a good baseline to train.


(2) breakout_standard_dqn.csv and the associated image are for the baseline DQN,
using spragnur's code, with settings consistent with NIPS except for the change
to 40000 steps per epoch.  (By doing this change I can make the code run in
under a day.) This is good for a baseline, though to be REALLY safe, I should
run the baseline using my modified version of spragnur's code, except it ignores
the human net.


(3) breakout_11-26-11-19_.csv and the corresponding files pertain to the revised
version of (1) here, they're designed to fix the three issues that prevented
full correctness of (1). Unfortunately, the results don't seem to be that much
better. =(


(4) breakout_11-26-16-57_.csv and the corresponding files pertain to (2), the
baseline DQN using my code instead of spragnur's, but it should be equivalent
to (2). Basically I'm getting the control experiment set up correctly.


(5) breakout_11-27-16-08_.csv and the corresponding files are something new: I
decided to simply ignore the training and have the testing be done only on my
classifier that was trained from game frames to actions. I set it to follow my
net 95% of the time and the random action 5% of the time (needed to get FIRE
actions to happen). The results are utter garbage. Hmmm ... perhaps I can better
investigate the distribution of actions to see what's happening? By the way, to
make this run fast, I simply set the number of steps in training to be 1, down
from 40000 (which was down from 50000 from the NIPS version).

NEW, let's do testing but with a fixed amount of episodes per iteration. For
this, I keep the testing steps to 10000 but ignore that during testing, and just
call run_episode 30 times. I think this should work, so long as any single
episode doesn't last for more than 10k steps (and for Breakout that shouldn't
happen).

(6) breakout_11-28-21-07_.csv and associated files are my method, with human net
plugged inside DQN, except I ran the appropriate 50k training steps per epoch. I
also did testing with a fixed 30 episodes per iteration.

(7) breakout_11-29-08-20_.csv, same as (6) except it's using the control method,
so default DQN.

-- NATURE --

(8) breakout_12-01-21-52_.csv an corresponding files are Breakout, default DQN, with
the NATURE settings.

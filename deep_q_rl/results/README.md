Here are the file names:

(1) breakout_11-25-20-26_.csv and log_11-25-20-26_.txt are the results and log
file, respectively, of DQN with human-boosted data, on the NIPS settings except
using 40000 steps per epoch, which is consistent with the baseline. This is NOT
supposed to be finalized, because it's missing the following:

- The human net was trained on sequences with frame skips 2, 3, and 4, but it
  should only be 4.
- The human net was trained on data which did NOT do the consecutive
  maximizations (to stop flickering effects).
- The human net was trained on data using my grayscale conversion, and not what
  the ALE would have returned. In fact, it's not even important that this
  matches what Google DeepMind did --- all that matters is that the conversion
  is the same as what SPRAGNUR'S code did, because his code is converting it to
  grayscale.

Despite these flaws, this still looks like a good baseline to trian.


(2) breakout_standard_dqn.csv and breakout_standard_dqn.png are the results file
and image, respectively, for the baseline DQN, using spragnur's code, with
settings consistent with NIPS except for the change to 40000 steps per epoch.
(By doing this change I can make the code run in under a day.) This is good for
a baseline, though to be REALLY safe, I should run the baseline using my
modified version of spragnur's code, except it ignores the human net.

The full_requirements.txt file has a dependency that requires you to install things in a very certain order.
For example, CVXOPT will require you to install CVXPY first, then CVXOPT.
Numpy is often required to install before other packages.
Trial and error will be needed to figure out a reliable install order.
Not to mention, the legacy OpenDR library is not compatible with anything above 3.7 it seems.
You will need to follow these instructions found here:
https://github.com/akanazawa/hmr/issues/82
Pasted:
Download Opendr from here: https://github.com/polmorenoc/opendr (this must be the 2019 version, issue reply was made in 2020)
[In order to download the 2019 version, you will need to go git clone, pick a commit back in 2019, check out, and zip the OpenDR directory.]
Extract opendr folder from archive downloaded from step 1. (opendr-master/opendr)
Zip again that opendr folder (opendr-master/opendr only not the all master folder) so that it will like opendr.zip
Then hit "Python -m pip install opendr.zip"
There are other methods, this is the one that worked for me.
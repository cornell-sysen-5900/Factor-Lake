#### Install Both
```
uv pip install bandit safety
```

#### Created Insecure File
```
#filename is bad_code.py

import subprocess

def bad():
    subprocess.call("ls", shell=True)

#hardcoded token/password in code
token = ";alkjdsalfjieajlh"

#manually working with files in temp directory is not safe

temp_dir = "/tmp"
```

#### Executed Bandit
```
#make sure you saved bad_code.py before executing this command

bandit bad_code.py

```

#### Executing Bandit with Options
```
#This doesn't change any metrics or anything, it only changes what it reports to you.

#sets logging level to "warning" - least verbose
bandit bad_code.py -l

#sets logging level to "info"
bandit bad_code.py -ll

#sets logging level to "debug" -  most verbose
bandit bad_code.py -lll

#scanning an entire directory
bandit -r .


```

#### NOSEC
```
#If you add the bandit code and prepend it with "nosec" as shown below, it will remove that scanning result from the metrics.


#hardcoded token/password in code
token = ";alkjdsalfjieajlh" # nosec: B105

#manually working with files in temp directory is not safe

temp_dir = "/tmp"
```
#### Ignoring Something Throughout Entire Program Instead of One by One with nosec
```
bandit bad_code.py -s B105
```

#### Safety 
```
#checks your project's dependencies for known security vulnerabilities

#to demonstrate its usage and value, I've installed a version of numpy that has a known security vulnerability.

uv pip install numpy==1.22.1

safety check

#to get more details
safety check --full-report
```
<img width="644" alt="image" src="https://github.com/user-attachments/assets/2b2f18ec-beb5-4319-bb8a-d9764324c0a7">

##### Full-Report Details
<img width="650" alt="image" src="https://github.com/user-attachments/assets/dc6bdadc-d80d-4d2e-8ce3-f130ad84247f">

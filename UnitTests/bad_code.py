import subprocess

def bad():
    subprocess.call("ls", shell=True)

# hardcoded token/password in code
token = ";alkjdsalfjieajlh"  # nosec: B105

# manually working with files in temp directory is not safe
temp_dir = "/tmp"
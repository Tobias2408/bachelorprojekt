import re
import toml

def increment_version(version):
    major, minor, patch = map(int, version.split('.'))
    patch += 1
    return f"{major}.{minor}.{patch}"

# Update version in pyproject.toml
with open("pyproject.toml", "r") as f:
    pyproject = toml.load(f)

current_version = pyproject["project"]["version"]
new_version = increment_version(current_version)
pyproject["project"]["version"] = new_version

with open("pyproject.toml", "w") as f:
    toml.dump(pyproject, f)

# Update version in setup.py
setup_file = "setup.py"
with open(setup_file, "r") as f:
    setup_content = f.read()

setup_content = re.sub(r'version="\d+\.\d+\.\d+"', f'version="{new_version}"', setup_content)

with open(setup_file, "w") as f:
    f.write(setup_content)

print(f"Version updated from {current_version} to {new_version}")

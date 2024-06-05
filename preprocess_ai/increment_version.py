import re
import toml

def increment_version(version):
    parts = list(map(int, version.split('.')))
    if len(parts) == 2:
        parts.append(0)  # Add patch version if missing
    parts[-1] += 1
    return ".".join(map(str, parts))

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

setup_content = re.sub(r'version="\d+\.\d+(\.\d+)?"', f'version="{new_version}"', setup_content)

with open(setup_file, "w") as f:
    f.write(setup_content)

print(f"Version updated from {current_version} to {new_version}")

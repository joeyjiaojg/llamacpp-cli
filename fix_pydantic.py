#!/usr/bin/env python3
"""Fix Pydantic V2 warnings by updating to ConfigDict."""

import re

# Read file
with open("src/llamacpp_cli/lb_proxy.py", "r") as f:
    content = f.read()

# Update import
content = content.replace(
    "from pydantic import BaseModel, Field",
    "from pydantic import BaseModel, ConfigDict, Field"
)

# Fix Field with example= to use json_schema_extra
content = re.sub(
    r'Field\((.*?), description="([^"]+)", example="([^"]+)"\)',
    r'Field(\1, description="\2", json_schema_extra={"example": "\3"})',
    content
)

# Fix class Config to model_config with ConfigDict
content = re.sub(
    r'    class Config:\n        json_schema_extra = \{([^}]+)\}',
    r'    model_config = ConfigDict(json_schema_extra={\1})',
    content,
    flags=re.MULTILINE | re.DOTALL
)

# Write back
with open("src/llamacpp_cli/lb_proxy.py", "w") as f:
    f.write(content)

print("✓ Fixed Pydantic V2 warnings")

#!/bin/bash

# link in vscode settings
rm /root/capsule/code/.vscode/Machine/settings.json
ln -s /root/vscode_machine_settings.json /root/capsule/code/.vscode/Machine/settings.json

# Cline MCP settings
CLINE_DIR="$HOME/capsule/code/.vscode/User/globalStorage/saoudrizwan.claude-dev/settings"
CLINE_CONFIG="$CLINE_DIR/cline_mcp_settings.json"

if [ ! -f "$CLINE_CONFIG" ]; then
  mkdir -p $CLINE_DIR
  cat > "$CLINE_CONFIG" << EOF
{
  "mcpServers": {
    "aind-metadata-mcp": {
      "command": "aind-metadata-mcp"
    }
  }
}
EOF
fi

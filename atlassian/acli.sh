export JIRA_SITE="amd-hub.atlassian.net"
export JIRA_EMAIL="nick.romero@amd.com"
read -rsp "Jira token: " JIRA_TOKEN; echo
printf '%s' "$JIRA_TOKEN" | acli jira auth login \
  --site "$JIRA_SITE" --email "$JIRA_EMAIL" --token
unset JIRA_TOKEN
acli jira auth status

# Sandbox Vault

This is a mock vault note for the local `sandbox-user`. It stands in for a real
Supabase `user-vaults` download during isolated/offline testing, so the
orchestrator's vault-sync step exercises the same downstream path without any S3
credentials or network access.

Add more `.md` files under `mock_vaults/sandbox-user/` to enrich local context.

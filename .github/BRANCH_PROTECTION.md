# Branch Protection Rules

This document outlines the recommended branch protection rules for the TinyLLM repository to ensure code quality and prevent accidental changes to critical branches.

## 🔒 Protection Rules for `master` and `main` Branches

### Required Status Checks

All of the following checks **must pass** before merging:

#### Core CI Checks
- ✅ **Lint & Type Check** (`ci.yml` - lint job)
- ✅ **Unit Tests** (`ci.yml` - test job) - All Python versions (3.11, 3.12, 3.13)
- ✅ **Integration Tests** (`ci.yml` - integration-test job)
- ✅ **Build Package** (`ci.yml` - build job)

#### Quality Gates
- ✅ **Quality Gate Summary** (`quality-gates.yml` - quality-summary job)
  - Coverage ≥80%
  - No linting errors
  - No type checking errors
  - No high/medium security issues
  - Test count ≥300
- ✅ **Dependency Freshness** (`quality-gates.yml` - dependency-freshness job)

#### Security Checks
- ✅ **Dependency Security Scan** (`security.yml` - dependency-scan job)
- ✅ **Code Security Analysis** (`security.yml` - code-security job)
- ✅ **Secret Detection** (`security.yml` - secret-scan job)

#### PR Validation
- ✅ **PR Metadata** (`pr-validation.yml` - pr-metadata job)
- ✅ **Test Plan Check** (`pr-validation.yml` - test-plan-check job)
- ✅ **Documentation Check** (`pr-validation.yml` - documentation-check job)
- ✅ **Conflict Check** (`pr-validation.yml` - conflict-check job)

#### Performance (Warning Only)
- ⚠️  **Performance Regression** (`performance-regression.yml`) - Warning if >10% slower, fails if >20% slower

### Merge Requirements

#### Pull Request Reviews
- **Required approving reviews:** 1
- **Dismiss stale reviews:** Yes (when new commits are pushed)
- **Require review from Code Owners:** Yes (if CODEOWNERS file exists)
- **Require approval of most recent reviewable push:** Yes

#### Commit Settings
- **Require linear history:** No (allow merge commits)
- **Require signed commits:** Recommended but not enforced
- **Require conversation resolution:** Yes (all conversations must be resolved)

#### Branch Restrictions
- **Restrict who can push:** Yes
  - Only allow repository administrators and specific users/teams
- **Allow force pushes:** No
- **Allow deletions:** No

### Additional Settings

#### Status Check Settings
- **Require branches to be up to date before merging:** Yes
  - Ensures PRs are tested against the latest code
- **Do not require status checks on creation:** No

#### Rules for Administrators
- **Include administrators:** Yes
  - Administrators must follow the same rules

## 🏷️ Protection Rules for Release Branches

For branches matching `release/*` or `v*`:

### Required Checks
- ✅ All CI checks from main branch
- ✅ Benchmark tests must pass
- ✅ No dependency vulnerabilities

### Additional Restrictions
- **Allow force pushes:** No
- **Allow deletions:** No
- **Require linear history:** Yes

## 🧪 Protection Rules for Development Branches

For branches matching `dev/*`, `feature/*`, `fix/*`:

### Required Checks (Lighter)
- ✅ Lint & Type Check
- ✅ Unit Tests (Python 3.11 only)
- ✅ PR Metadata validation

### Settings
- **Required reviews:** 0 (for personal development branches)
- **Allow force pushes:** Yes (for personal branches only)

## 📋 Setting Up Branch Protection

### GitHub UI

1. Go to **Settings** → **Branches** → **Add branch protection rule**

2. **Branch name pattern:** `master` (or `main`)

3. Enable the following options:

   **Protect matching branches:**
   - [x] Require a pull request before merging
     - [x] Require approvals: 1
     - [x] Dismiss stale pull request approvals when new commits are pushed
     - [x] Require review from Code Owners
     - [x] Require approval of the most recent reviewable push
     - [x] Require conversation resolution before merging

   - [x] Require status checks to pass before merging
     - [x] Require branches to be up to date before merging
     - Select all required status checks listed above

   - [x] Require signed commits (optional but recommended)

   - [x] Require linear history (optional)

   - [x] Include administrators

   - [x] Restrict who can push to matching branches
     - Add: Repository administrators

   - [x] Allow force pushes: Never

   - [x] Allow deletions: No

4. Click **Create** or **Save changes**

### GitHub CLI

```bash
# Protect master branch
gh api repos/{owner}/{repo}/branches/master/protection \
  --method PUT \
  --field required_pull_request_reviews[required_approving_review_count]=1 \
  --field required_pull_request_reviews[dismiss_stale_reviews]=true \
  --field required_pull_request_reviews[require_code_owner_reviews]=true \
  --field required_status_checks[strict]=true \
  --field required_status_checks[contexts][]="Lint & Type Check" \
  --field required_status_checks[contexts][]="Unit Tests (Python 3.11)" \
  --field required_status_checks[contexts][]="Integration Tests (Python 3.11)" \
  --field required_status_checks[contexts][]="Build Package" \
  --field required_status_checks[contexts][]="Quality Gate Summary" \
  --field enforce_admins=true \
  --field required_conversation_resolution=true \
  --field restrictions=null
```

### Terraform Configuration

```hcl
resource "github_branch_protection" "master" {
  repository_id = github_repository.repo.node_id
  pattern       = "master"

  required_pull_request_reviews {
    required_approving_review_count = 1
    dismiss_stale_reviews           = true
    require_code_owner_reviews      = true
  }

  required_status_checks {
    strict = true
    contexts = [
      "Lint & Type Check",
      "Unit Tests (Python 3.11)",
      "Unit Tests (Python 3.12)",
      "Unit Tests (Python 3.13)",
      "Integration Tests (Python 3.11)",
      "Integration Tests (Python 3.12)",
      "Build Package",
      "Quality Gate Summary",
      "Dependency Security Scan",
      "Code Security Analysis",
      "Secret Detection",
      "PR Metadata",
    ]
  }

  enforce_admins              = true
  require_signed_commits      = false
  require_conversation_resolution = true

  restrict_pushes {
    blocks_creations = false
  }
}
```

## 🔍 Monitoring and Maintenance

### Regular Review Schedule

**Monthly:**
- Review failed PRs and identify common issues
- Update status check requirements if workflows change
- Verify all required checks are still relevant

**Quarterly:**
- Review and update protection rules based on team feedback
- Assess if coverage/quality thresholds need adjustment
- Review administrator exceptions log

**After Major Changes:**
- Update protection rules when CI workflows are modified
- Add new required checks for new critical workflows
- Remove deprecated status checks

### Troubleshooting

#### Common Issues

**1. PR Can't Merge Due to Status Check**
- Ensure all workflows completed successfully
- Check for skipped workflows (may not count as passed)
- Verify branch is up to date with base branch

**2. Administrator Can't Bypass Rules**
- Check "Include administrators" setting
- Verify you have the "repository administrator" role

**3. Status Check Not Appearing**
- Workflow may not have run yet
- Check if workflow is triggered by `pull_request` event
- Verify workflow job names match required check names

## 📚 Additional Resources

- [GitHub Branch Protection Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [Status Checks Documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/about-status-checks)
- [Required Reviews Documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/about-pull-request-reviews)

---

*Last updated: 2025-12-23*
*This document should be updated whenever CI/CD workflows change.*

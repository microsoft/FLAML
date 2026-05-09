# How to merge FLAML Github commits into FLAML-Internal

- [How to merge FLAML Github commits into FLAML-Internal](#how-to-merge-flaml-github-commits-into-flaml-internal)
  - [Step 1. Get status of current FLAML-Internal](#step-1-get-status-of-current-flaml-internal)
  - [Step 2. Get the latest github commit id](#step-2-get-the-latest-github-commit-id)
  - [Step 3. Cherry-pick github commits into FLAML-Internal](#step-3-cherry-pick-github-commits-into-flaml-internal)
    - [Fix conflicts](#fix-conflicts)
    - [Remove files which should be removed](#remove-files-which-should-be-removed)
  - [Step 4. Commit and raise a PR](#step-4-commit-and-raise-a-pr)

## Step 1. Get status of current FLAML-Internal

Find the last merged github commit id in the last `merge github till <commit_id>` PR:

<div align="center"><img src="https://raw.githubusercontent.com/thinkall/imgbed/master/img/202407221649515.png" height="350"/></img></div>
`165d746` in the example.

## Step 2. Get the latest github commit id

<div align="center"><img src="https://raw.githubusercontent.com/thinkall/imgbed/master/img/202407221647162.png" height="350"/></img></div>
`67f4048` in the example.

## Step 3. Cherry-pick github commits into FLAML-Internal

Go to the folder `FLAML-Internal` and run below commands:

```
git remote add ms https://github.com/microsoft/FLAML.git
git fetch --all
git checkout main
git pull origin main
git checkout -b merge_github
git cherry-pick 165d746...67f4048
```

The output could be like:

```
Auto-merging .gitignore
CONFLICT (content): Merge conflict in .gitignore
Auto-merging flaml/automl/automl.py
CONFLICT (content): Merge conflict in flaml/automl/automl.py
Auto-merging flaml/automl/model.py
Auto-merging notebook/autogen_agentchat_RetrieveChat.ipynb
Auto-merging setup.py
CONFLICT (content): Merge conflict in setup.py
Auto-merging test/automl/test_classification.py
Auto-merging test/automl/test_forecast.py
CONFLICT (content): Merge conflict in test/automl/test_forecast.py
Auto-merging test/automl/test_notebook_example.py
Auto-merging website/yarn.lock
error: could not apply d8129b92... Fix typos, upgrade yarn packages, add some improvements (#1290)
hint: After resolving the conflicts, mark them with
hint: "git add/rm <pathspec>", then run
hint: "git cherry-pick --continue".
hint: You can instead skip this commit with "git cherry-pick --skip".
hint: To abort and get back to the state before "git cherry-pick",
hint: run "git cherry-pick --abort".
```

### Fix conflicts

Check the conflicted files one by one, you can fix the conflicts with the help of VSCode.

<div align="center"><img src=https://raw.githubusercontent.com/thinkall/imgbed/master/img/202305301523327.png height=600></img></div>

### Remove files which should be removed

**Notice: Files removed in github will not be removed in internal version and will not raise a conflict.** Thus we should check the files added in internal version and remove those should be removed, i.e., those removed in github version.

```
git diff --name-only --diff-filter=D github_commit_id_last_merge github_commit_id_latest
# git diff --name-only --diff-filter=D 165d746 67f4048 > files_to_remove.txt
```

The output is like below:

```
.flake8
docs/Makefile
```

Therefore, We know that we should also remove these files from internal version.

Run below command to remove the files.

```
while IFS= read -r file; do rm "$file"; done < files_to_remove.txt
rm files_to_remove.txt
```

**Note**

These files from github repo are intended to be removed in our internal repo:

- `test/pipeline_tuning_example`
- `flaml/automl/spark/configs.py`

Or you can get a removed list of our internal repo by:

```
git diff --name-only --diff-filter=D internal_commit_id_before_last_merge internal_commit_id_latest
# git diff --name-only --diff-filter=D 7d6a3b63 4b685424 > files_to_remove.txt
```

## Step 4. Commit and raise a PR

Once all conflicts are solved, commit the changes and raise a PR to FLAML-Internal.

```
git add .
git cherry-pick --continue
git push origin merge_github
```

Add commit messages in the PR.

<div align="center"><img src=https://raw.githubusercontent.com/thinkall/imgbed/master/img/202305301531370.png height=350></img></div>

And **REMOVE all `#` to avoid linking wrong work items.**

<div align="center"><img src=https://raw.githubusercontent.com/thinkall/imgbed/master/img/202305301542597.png height=300></img></div>

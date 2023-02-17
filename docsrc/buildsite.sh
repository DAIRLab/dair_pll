#!/bin/bash
set -x

pwd ls -lah
git log -1 --pretty=%ct
export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)

##############
# BUILD DOCS #
##############
 
# build tools
pip install --upgrade pip setuptools wheel

# git deps
pip install git+https://github.com/mshalm/sappy.git
pip install git+https://github.com/DAIRLab/drake-pytorch.git
# install package to get deps
pip install -e .

# build docs from source
python docsrc/build_docs.py --regenerate_deps

#######################
# Update GitHub Pages #
#######################

git config --global user.name "${GITHUB_ACTOR}"
git config --global user.email "${GITHUB_ACTOR}@users.noreply.github.com"
 
docroot=`mktemp -d`
rsync -av "docs/" "${docroot}/"
 
pushd "${docroot}"

git init
git remote add deploy "https://token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
git checkout -b gh-pages
 
# Adds .nojekyll file to the root to signal to GitHub that  
# directories that start with an underscore (_) can remain
touch .nojekyll
 
# Add README
cat > README.md <<EOF
# README for the GitHub Pages Branch
This branch is simply a cache for the website served from https://dairlab.github.io/dair_pll/,
and is not intended to be viewed on github.com.

For more information on how this site is built using Sphinx, Read the Docs, and GitHub Actions/Pages, see:
 * https://www.docslikecode.com/articles/github-pages-python-sphinx/
 * https://tech.michaelaltfield.net/2020/07/18/sphinx-rtd-github-pages-1
 * 
EOF
 
# Copy the resulting html pages built from Sphinx to the gh-pages branch 
git add .
 
# Make a commit with changes and any new files
msg="Updating Docs for commit ${GITHUB_SHA} made on `date -d"@${SOURCE_DATE_EPOCH}" --iso-8601=seconds` from ${GITHUB_REF} by ${GITHUB_ACTOR}"
git commit -am "${msg}"
 
# overwrite the contents of the gh-pages branch on our github.com repo
git push deploy gh-pages --force
 
popd # return to main repo sandbox root
 
# exit cleanly
exit 0

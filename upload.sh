#!/bin/bash
branch_name=$(git symbolic-ref --short HEAD)
commit=$1
git add .
git commit -m "$commit"
git push origin "$branch_name"	

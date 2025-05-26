# run:
# `just sync`
# --filter ':- .gitignore' enables to mimic behavior of gitingore on exluding patterns for rsync

sync:
    rsync -rah . \
        --verbose \
        --exclude ".git" \
        --exclude "presentations" \
        --filter ':- .gitignore' \
        --rsh "ssh -p 44422" \
        hse_student@46.229.141.80:~/a.soldatov

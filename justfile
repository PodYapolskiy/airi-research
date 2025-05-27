# run:
# `just sync`
# --filter ':- .gitignore' enables to mimic behavior of gitingore on exluding patterns for rsync
# `just get_data /home/HDD12TB/datasets/images/emotions/ACMMM25/AVI/`

sync:
    rsync -rah . \
        --verbose \
        --exclude ".git" \
        --exclude "presentations" \
        --filter ':- .gitignore' \
        --rsh "ssh -p 44422" \
        hse_student@46.229.141.80:~/a.soldatov

get_data path:
    scp -r \
        -P 44422 \
        hse_student@46.229.141.80:{{ path }} \
        ./data

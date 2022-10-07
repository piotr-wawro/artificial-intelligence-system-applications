COUNT=$(env | grep VIRTUAL_ENV | wc -l)

if [[ $COUNT == 0 ]]
    then
    echo "Cannot install requirements. Python virtual environment not activated."
    exit 1
fi

if [[ "$1" == "linux" ]]
    then
elif [[ "$1" == "windows" ]]
    then
else
    echo "Cannot install system specific requirements. Unknown operating system."
    exit 1
fi

pip install -r ./requirements/pypi.txt
pip freeze > ./requirements/lock.txt

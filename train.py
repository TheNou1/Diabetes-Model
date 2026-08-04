from comparison import compare

print("DIRTY DATASET")

dirty_results = compare(
    x_train,
    x_test,
    y_train,
    y_test
)

print(dirty_results)

print()

print("CLEAN DATASET")

clean_results = compare(
    xc_train,
    xc_test,
    yc_train,
    yc_test
)

print(clean_results)
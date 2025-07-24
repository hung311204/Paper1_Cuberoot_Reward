






### DON"T RUN THIS TWICE

import grid2op
env_name = "l2rpn_case14_sandbox"  # or any other...
env = grid2op.make(env_name)

# extract 1% of the "chronics" to be used in the validation environment. The other 1% will
# be used for test
nm_env_train, nm_env_val, nm_env_test = env.train_val_split_random(
    pct_val=2., pct_test=2., add_for_val="val", 
    add_for_test="test"
)

# and now you can use the training set only to train your agent:
print(f'The name of the training environment is "{nm_env_train}"')
print(f'The name of the validation environment is "{nm_env_val}"')
print(f'The name of the test environment is "{nm_env_test}"')


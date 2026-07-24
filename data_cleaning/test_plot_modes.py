import pandas as pd
import analysis as an

# simple df with numeric index
df = pd.DataFrame({'a':[1,2,3,4,5]}, index=[0,1,2,3,4])
# per-sample movement_modes (len == len(df)-1)
modes = ['cat','cow','cat','cow']

# should not raise
an.plot_movement_modes(df, modes)
print('list OK')

# DataFrame style from sense_movement_mode
mdf = pd.DataFrame({'start_time':[1,3], 'mode':['cat','cow']})
# should not raise
an.plot_movement_modes(df, mdf)
print('df OK')

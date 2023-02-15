'''
Nick Smith
Algorithm for determing whether a team should go for a blocked punt or a big return
Developed for Sports Analytics and Research Club at Johns Hopkins Fall 2022
Project Members: Nick Smith, Owen Hartman, Drew Amunategui
'''

##reading in data and assigning to certain data points
data = input("Enter each data value with a space in between: \n seconds left in game \n precipitation level (0: none, 1: moderate, 2: heavy)\n cold/wind (0:neither, 1: one of them, 2:both)\n points trailing by \n field position \n")
data_array = data.split(' ')
seconds_left = int(data_array[0])
precipitation = int(data_array[1])
cold_wind = int(data_array[2])
deficit = int(data_array[3])
field_position = int(data_array[4])
possesions_needed = deficit/8


## find blocking situation
if seconds_left <= 30:
    ## less than 30 seconds must block situation
    print("Attempt a block")
elif seconds_left <= 60 and deficit > 3:
    ## less than 60 and need more than a fg to win
    print("Attempt a punt block")
elif field_position <= 5:
    ## punt team backed up. guaranteed decent field position even if punt gets off
    print("Attempt a punt block")
elif precipitation == 2 or cold_wind == 2:
    ## heavy rain (punt operation time slower
    print("Attempt a punt block")
elif seconds_left < possesions_needed * 120:
    ##arbitrary number. but thought is to include time due to having to play defense or onside kick
    print("Attempt a punt block")
else:
    print("Set up a return")

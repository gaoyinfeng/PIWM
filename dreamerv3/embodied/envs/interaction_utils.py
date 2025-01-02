import numpy as np


# isim coordinate frame transfer
def localize_transform_list(origin_location, origin_heading, global_location_list, global_heading_list=None): # is used for making prediciton data
  localized_location_list = []
  localized_heading_list = []
  for index in range(len(global_location_list)):
    # first find out zero padding predicition state
    if global_location_list[index][0] == 0. and global_location_list[index][1] == 0.:
      if localized_location_list:
        localized_location_list.append(localized_location_list[-1])
        continue
    # then localize the location
    if global_heading_list is not None:
      localized_location, localized_heading = localize_transform(origin_location, origin_heading, global_location_list[index], global_heading_list[index])
      localized_heading_list.append(localized_heading)
    else:
      localized_location, _ = localize_transform(origin_location, origin_heading, global_location_list[index])
    localized_location_list.append(localized_location)
  
  return localized_location_list, localized_heading_list

def localize_vector_transform_list(origin_location, origin_heading, global_vector_transform_list): # is used for vector state input
  localized_vector_transform_list = []
  for index in range(len(global_vector_transform_list)):
    vector_transform = global_vector_transform_list[index]
    global_location_prev, global_location, global_heading = vector_transform[0:2], vector_transform[2:4], vector_transform[4]
    localized_location_prev, _ = localize_transform(origin_location, origin_heading, global_location_prev)
    localized_location, localized_heading = localize_transform(origin_location, origin_heading, global_location, global_heading)

    localized_vector_transform_list.append(localized_location_prev + localized_location + [localized_heading])
  
  return localized_vector_transform_list

def localize_transform(origin_location, origin_heading, global_location, global_heading=None):
  localized_x = (global_location[1] - origin_location[1])*np.cos(origin_heading) - (global_location[0] - origin_location[0])*np.sin(origin_heading)
  localized_y = (global_location[1] - origin_location[1])*np.sin(origin_heading) + (global_location[0] - origin_location[0])*np.cos(origin_heading)
  localized_location = [localized_x, localized_y]
  if global_heading is not None:
    localized_heading = global_heading - origin_heading
  else:
    localized_heading = None

  return localized_location, localized_heading

# def delocalize_transform(origin_location, origin_heading, localized_location, localized_heading=None):
#   global_x = localized_location[1] * np.cos(origin_heading) - localized_location[0] * np.sin(origin_heading) + origin_location[0]
#   global_y = localized_location[1] * np.sin(origin_heading) + localized_location[0] * np.cos(origin_heading) + origin_location[1]
#   global_location = [global_x, global_y]
#   if localized_heading is not None:    
#     global_heading = localized_heading + origin_heading
#   else:
#     global_heading = None

#   return global_location, global_heading

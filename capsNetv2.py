

#####
#   Improve last model
#   a) Dimensions of u and v needn't be same for dot product
#   b) Using seperate W for each cell in v
#   c) Using tf.split instead of slicing operations
#   d) Use newer tf versions
#   e) Design class structure to accomodate multiple layers
#   f) avoid multiple reshape operations for matmul and tile for mul
#   g) Not keeping multiple dimensions between.
#  ########
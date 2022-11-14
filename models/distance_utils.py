import torch

def tile_first_dim(x, first_dim_multiples):
	shape_list = x.shape.as_list()
	num_first_dim = len(first_dim_multiples)
	if num_first_dims > len(shape_list):
		raise ValueError("Number of first dimensions must not be greater than the total dimensionality of input tensor: %d vs. %d." %
										 (num_first_dim, len(shape_list)))
	num_other_dims = len(shape_list) - num_first_dim
	multiples = list(first_dim_multiples) + [1] * num_other_dims

	return torch.tile(x, tuple(multiples))

def compute_distance_matrix(start_points,
														end_points,
														distance_fn):
	def expand_and_tile_axis_01(x, target_axis, target_dim):
		if target_axis not in [0, 1]:
			raise ValueError('Only supports 0 or 1 as target axis: %s.' % str(target_axis))
		
		x = torch.unsqueeze(x, dim=target_axis)
		first_dim_multiples = [1, 1]
		first_dim_multiples[target_axis] = target_dim
		return tile_first_dim(x, first_dim_multiples=first_dim_multiples)

	num_start_points = start_points.size(dim=0)
	num_end_points = start_points.size(dim=0)

	start_points = expand_and_tile_axis_01(
		start_points, target_axis=1, target_dim=num_end_points)
	end_points = expand_and_tile_axis_01(
		end_points, target_axis=0, target_dim=num_start_points)

	return distance_fn(start_points, end_points)

	
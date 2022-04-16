class Parameters():
    # How many values minimum must each attribute take
    example_umbral = 0
    # Ganancia de la informacion min limit
    gain_umbral = 0
    # Maximum height the tree can be
    height_limit = 20

    # Random Forest Variables
    # Random amount of attributes to consider in RF
    split_attr_limit = None    
    # Amount of data in each RF sample, percentage wise
    sample_size = 0.9
    # How many samples to use in RF
    number_samples = 10

    def set_example_umbral(example_umbral):
        Parameters.example_umbral = int(example_umbral)
    
    def set_gain_umbral(gain_umbral):
        Parameters.gain_umbral = float(gain_umbral)
    
    def set_height_limit(height_limit):
        Parameters.height_limit = int(height_limit)
    
    def set_split_attr_limit(split_attr_limit):
        Parameters.split_attr_limit = int(split_attr_limit)
    
    def set_sample_size(size):
        Parameters.sample_size = float(size)
    
    def set_number_samples(amount):
        Parameters.number_samples = int(amount)

    def get_example_umbral():
        return Parameters.example_umbral
        
    def get_gain_umbral():
        return Parameters.gain_umbral
        
    def get_height_limit():
        return Parameters.height_limit
        
    def get_split_attr_limit():
        return Parameters.split_attr_limit

    def get_sample_size():
        return Parameters.sample_size

    def get_number_samples():
        return Parameters.number_samples

    def set_random_forest_values(example_umbral, gain_umbral, height_limit, split_attr_limit, sample_size, number_size):
        if example_umbral is not None: Parameters.set_example_umbral(example_umbral)
        if gain_umbral is not None: Parameters.set_gain_umbral(gain_umbral)
        if height_limit is not None: Parameters.set_height_limit(height_limit)
        if split_attr_limit is not None: Parameters.set_split_attr_limit(split_attr_limit)
        if sample_size is not None: Parameters.set_sample_size(sample_size)
        if number_size is not None: Parameters.set_number_samples(number_size)
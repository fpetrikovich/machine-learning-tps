class Parameters():
    amount_of_points = 25
    margin = 0.5
    misclassifications = 1

    def set_amount_of_points(amount_of_points):
        Parameters.amount_of_points = int(amount_of_points)

    def set_misclassifications(misclassifications):
        Parameters.misclassifications = int(misclassifications)

    def set_margin(margin):
        Parameters.margin = float(margin)

    def get_misclassifications():
        return Parameters.misclassifications

    def get_margin():
        return Parameters.margin

    def get_amount_of_points():
        return Parameters.amount_of_points

    def set_values(amount_of_points, misclassifications):
        if amount_of_points is not None: Parameters.set_amount_of_points(amount_of_points)
        if margin is not None: Parameters.set_margin(margin)
        if misclassifications is not None: Parameters.set_misclassifications(misclassifications)

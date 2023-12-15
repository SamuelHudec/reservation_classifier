# I like to hardcode column names
CATEGORICAL_FEATURES = ["hotel", "arrival_date_month", "market_segment", "deposit_type", "customer_type"]
NUMERIC_FEATURES = [
    "lead_time",
    "stays_in_nights",
    "adults",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "agent",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

TRAINING_COLUMNS = CATEGORICAL_FEATURES + [
    "lead_time",
    "stays_in_week_nights",
    "stays_in_weekend_nights",
    "adults",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "agent",
    "required_car_parking_spaces",
    "total_of_special_requests",
]
RESPONSE_COLUMN = "is_canceled"

NUMERIC_IMPUTER_STRATEGY = "mean"
CATEGORICAL_IMPUTER_STRATEGY = "unknown"

RANDOM_SEED = 42
TEST_SIZE = 0.2

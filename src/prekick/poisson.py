from math import exp, factorial


def poisson_goal_probability(expected_goals: float, goals: int) -> float:
    """Return the probability of scoring exactly `goals` goals."""

    if expected_goals < 0:
        raise ValueError("expected_goals must be non-negative")

    if goals < 0:
        raise ValueError("goals must be non-negative")

    return (
        exp(-expected_goals)
        * expected_goals**goals
        / factorial(goals)
    )

from enum import Enum
from typing import Tuple

class CostType(Enum):
    CONSTANT = "constant"
    # noam noya comment : complexity
    PRICE_RESPONSIVE = "price_responsive"

    # Future types can be added here

def calculate_costs(cost_type: CostType, config: dict) -> Tuple[float, float]:
    """
    Calculate reserve and dispatch costs based on type and configuration
    
    Args:
        cost_type: Type of cost structure to use
        config: Configuration dictionary containing cost parameters
        
    Returns:
        Tuple[float, float]: (reserve_price, dispatch_price)
    """
    if cost_type == CostType.CONSTANT:
        return (
            config.get('reserve_price', 20.0),
            config.get('dispatch_price', 10.0)
        )
    # elif cost_type == CostType.PRICE_RESPONSIVE:
    else:
        raise ValueError(f"Unknown cost type: {cost_type}")

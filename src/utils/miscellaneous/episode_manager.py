######################################################
#                                                    #
#                       EPISODE                      #
#                       MANAGER                      #
#                                                    #
######################################################


"""
Episode Manager Module

Provides convert_numpy_types() for JSON serialization of objects containing numpy types.
"""




######################################################
#                                                    #
#                      LIBRARIES                     #
#                                                    #
######################################################


import json
import numpy as np




######################################################
#                                                    #
#                      FUNCTIONS                     #
#                                                    #
######################################################


def convert_numpy_types(obj):
    """
    Recursively convert numpy types and other non-serializable objects to Python native types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types or other non-serializable objects
        
    Returns:
        Object with all non-serializable types converted to Python native types
    """
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_types(item) for item in obj.tolist()]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Handle objects with __dict__ (like LogprobsResult)
        try:
            return convert_numpy_types(obj.__dict__)
        except:
            return str(obj)
    elif hasattr(obj, 'as_dict'):
        # Handle objects with as_dict() method
        try:
            return convert_numpy_types(obj.as_dict())
        except:
            return str(obj)
    else:
        # For other types, try to convert to string as fallback
        try:
            json.dumps(obj)  # Test if it's already serializable
            return obj
        except (TypeError, ValueError):
            return str(obj)




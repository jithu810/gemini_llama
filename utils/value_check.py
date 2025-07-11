def safe_int(value, default):
        """
        Converts the given value to an integer, returning a default value if conversion fails.
        Args:
            value: The value to convert to an integer.
            default: The value to return if conversion fails due to ValueError or TypeError.
        Returns:
            int: The converted integer value, or the default if conversion is unsuccessful.
        """
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
        
def safe_float(value, default):
    """
    Converts the given value to a float, returning a default value if conversion fails.
    Args:
        value: The value to convert to float.
        default: The value to return if conversion fails due to ValueError or TypeError.
    Returns:
        float: The converted float value, or the default if conversion is unsuccessful.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
    
    
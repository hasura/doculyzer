import enum
import logging
import re
from datetime import datetime, time
from typing import Optional, Tuple, Union

from doculyzer.document_parser.lru_cache import ttl_cache

logger = logging.getLogger(__name__)


class TemporalType(enum.Enum):
    """Enumeration for different types of temporal data."""
    NONE = 0  # Not a temporal string
    DATE = 1  # Date only (no time component)
    TIME = 2  # Time only (no date component)
    DATETIME = 3  # Combined date and time
    TIME_RANGE = 4  # Time range (start and end times)


@ttl_cache(maxsize=256, ttl=3600)
def detect_temporal_type(input_string: str) -> TemporalType:
    """
    Detect if a string represents a date, time, datetime, time range, or none of these.

    Args:
        input_string: String to analyze

    Returns:
        TemporalType enum indicating the type of temporal data
    """
    try:
        # Import dateutil parser
        from dateutil import parser

        # Check if it's an obviously non-temporal string
        if not input_string or not isinstance(input_string, str):
            return TemporalType.NONE

        # If the string is very long or has many words, it's probably not a date/time
        if len(input_string) > 50 or len(input_string.split()) > 8:
            return TemporalType.NONE

        # Check for time range patterns first
        time_range_patterns = [
            r'^\d{1,2}:\d{2}\s*[-–—to]\s*\d{1,2}:\d{2}$',  # 14:00-16:00, 2:00-4:00
            r'^\d{1,2}:\d{2}\s*(?:am|pm)\s*[-–—to]\s*\d{1,2}:\d{2}\s*(?:am|pm)$',  # 2:00pm-4:00pm, 9:00am-5:00pm
            r'^\d{1,2}\s*(?:am|pm)\s*[-–—to]\s*\d{1,2}\s*(?:am|pm)$',  # 2pm-4pm, 9am-5pm
            r'^\d{1,2}[-–—]\d{1,2}\s*(?:am|pm)$',  # 2-4pm, 9-11am
        ]

        for pattern in time_range_patterns:
            if re.match(pattern, input_string.lower().strip(), re.IGNORECASE):
                return TemporalType.TIME_RANGE

        # Check if it's a time-only string (no date component)
        time_patterns = [
            r'^\d{1,2}:\d{2}(:\d{2})?(\s*[ap]\.?m\.?)?$',  # 3:45pm, 15:30, 3:45:30 PM
            r'^\d{1,2}\s*[ap]\.?m\.?$',  # 3pm, 11 a.m.
            r'^([01]?\d|2[0-3])([.:][0-5]\d)?([.:][0-5]\d)?$',  # 0500, 13.45, 22:30:15
            r'^(noon|midnight)$'  # noon, midnight
        ]

        for pattern in time_patterns:
            if re.match(pattern, input_string.lower().strip()):
                return TemporalType.TIME

        # Try to parse with dateutil
        try:
            result = parser.parse(input_string, fuzzy=False)

            # Check if it has a non-default time component
            # Default time is usually 00:00:00
            has_time = (result.hour != 0 or result.minute != 0 or result.second != 0 or
                        'am' in input_string.lower() or 'pm' in input_string.lower() or
                        ':' in input_string)

            # If the input string contains typical time separators (:) or indicators (am/pm)
            # even if the parsed time is 00:00:00, consider it a datetime
            if has_time:
                return TemporalType.DATETIME
            else:
                return TemporalType.DATE

        except (parser.ParserError, ValueError):
            # If dateutil couldn't parse it, it's likely not a date/time
            return TemporalType.NONE

    except Exception as e:
        logger.warning(f"Error in detect_temporal_type: {str(e)}")
        return TemporalType.NONE


@ttl_cache(maxsize=128, ttl=3600)
def parse_time_range(time_range_str: str) -> Tuple[Optional[Union[time, datetime]], Optional[Union[time, datetime]]]:
    """
    Parse a time range string into start and end time objects.

    Args:
        time_range_str: String representing a time range (e.g., "14:00-16:00")

    Returns:
        Tuple of (start_time, end_time) as time or datetime objects
    """
    try:
        from dateutil import parser

        # Normalize the separator
        normalized = re.sub(r'[-–—to]+', '-', time_range_str)

        # Check if the range uses a dash with AM/PM at the end (e.g., "9-5pm")
        am_pm_end_match = re.match(r'(\d{1,2})[-–—](\d{1,2})\s*([ap]\.?m\.?)', normalized, re.IGNORECASE)
        if am_pm_end_match:
            start_hour = int(am_pm_end_match.group(1))
            end_hour = int(am_pm_end_match.group(2))
            am_pm = am_pm_end_match.group(3).lower()

            # Adjust for AM/PM
            if 'p' in am_pm and end_hour < 12:
                end_hour += 12
                # If end is PM, and start hour is less than end hour before adjustment,
                # then start is also PM
                if start_hour < end_hour - 12:
                    start_hour += 12

            start_time = time(hour=start_hour)
            end_time = time(hour=end_hour)
            return start_time, end_time

        # Split on the separator
        parts = normalized.split('-')
        if len(parts) != 2:
            logger.warning(f"Could not split time range '{time_range_str}' into exactly two parts")
            return None, None

        start_str, end_str = parts

        # If the end has AM/PM but start doesn't, and they're both simple hours
        start_simple = re.match(r'^\d{1,2}$', start_str.strip())
        end_has_ampm = re.search(r'[ap]\.?m\.?', end_str, re.IGNORECASE)

        if start_simple and end_has_ampm and not re.search(r'[ap]\.?m\.?', start_str, re.IGNORECASE):
            # Add the AM/PM from the end to the start as well, for consistency
            am_pm = re.search(r'([ap]\.?m\.?)', end_str, re.IGNORECASE).group(1)
            start_str = f"{start_str} {am_pm}"

        # Parse both parts
        try:
            start_time = parser.parse(start_str).time()
        except (ValueError, parser.ParserError):
            logger.warning(f"Could not parse start time '{start_str}'")
            start_time = None

        try:
            end_time = parser.parse(end_str).time()
        except (ValueError, parser.ParserError):
            logger.warning(f"Could not parse end time '{end_str}'")
            end_time = None

        return start_time, end_time

    except Exception as e:
        logger.warning(f"Error parsing time range '{time_range_str}': {str(e)}")
        return None, None


@ttl_cache(maxsize=128, ttl=3600)
def create_semantic_time_range_expression(time_range_str: str) -> str:
    """
    Convert a time range string into a rich semantic natural language expression.

    Args:
        time_range_str: String representing a time range (e.g., "14:00-16:00")

    Returns:
        A natural language representation of the time range with rich semantic context
    """
    try:
        start_time, end_time = parse_time_range(time_range_str)

        if not start_time or not end_time:
            return time_range_str  # Return original if parsing failed

        # Generate semantic expressions for both times
        start_semantic = create_semantic_time_expression(start_time)
        end_semantic = create_semantic_time_expression(end_time)

        # Combine them
        return f"from {start_semantic} until {end_semantic}"

    except Exception as e:
        logger.warning(f"Error creating semantic time range expression: {str(e)}")
        return time_range_str  # Return original on error


@ttl_cache(maxsize=128, ttl=3600)
def create_semantic_time_expression(time_obj):
    """
    Convert a time object into a rich semantic natural language expression.

    Args:
        time_obj: A datetime or time object containing time information

    Returns:
        A natural language representation of the time with rich semantic context
    """
    try:
        # Extract hour and minute from either a datetime or time object
        if hasattr(time_obj, 'hour'):
            hour = time_obj.hour
        else:
            # Fallback for unknown object type
            return str(time_obj)

        if hasattr(time_obj, 'minute'):
            minute = time_obj.minute
        else:
            minute = 0

        if hasattr(time_obj, 'second'):
            second = time_obj.second
        else:
            second = 0

        if hasattr(time_obj, 'microsecond'):
            microsecond = time_obj.microsecond
        else:
            microsecond = 0

        # Determine AM/PM
        am_pm = "AM" if hour < 12 else "PM"

        # Convert to 12-hour format
        hour_12 = hour % 12
        if hour_12 == 0:
            hour_12 = 12

        # Time of day label
        if 4 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"

        # Determine quarter of hour
        quarter_labels = {0: "o'clock", 15: "quarter past", 30: "half past", 45: "quarter to"}

        # Default time description
        time_desc = f"{hour_12}:{minute:02d} {am_pm}"

        # Explicit minute description
        minute_desc = f", at minute {minute}" if minute != 0 else ""

        # Check for special times
        if minute in quarter_labels and second == 0:
            if minute == 45:
                next_hour = (hour_12 % 12) + 1
                if next_hour == 0:
                    next_hour = 12
                time_desc = f"{quarter_labels[minute]} {next_hour} {am_pm}"
            else:
                time_desc = f"{quarter_labels[minute]} {hour_12} {am_pm}"

        # Create full semantic expression
        result = f"at {time_desc} in the {time_of_day}{minute_desc}"

        # Add seconds if non-zero
        if second > 0 or microsecond > 0:
            if microsecond > 0:
                result += f", at second {second}.{microsecond // 1000:03d}"
            else:
                result += f", at second {second}"

        return result

    except Exception as e:
        logger.warning(f"Error converting time to semantic expression: {str(e)}")
        return str(time_obj)  # Return string representation on error


@ttl_cache(maxsize=128, ttl=3600)
def create_semantic_date_expression(date_str: str) -> str:
    """
    Convert a date string into a rich semantic natural language expression.

    Args:
        date_str: A string representing a date in various possible formats

    Returns:
        A natural language representation of the date with rich semantic context
    """
    try:
        # Import dateutil parser
        from dateutil import parser

        # Parse the date string using dateutil's flexible parser
        parsed_date = parser.parse(date_str)

        # Check if this is a datetime with significant time component
        has_time = False
        if hasattr(parsed_date, 'hour') and hasattr(parsed_date, 'minute'):
            if parsed_date.hour != 0 or parsed_date.minute != 0 or parsed_date.second != 0:
                has_time = True

        # If this has a significant time component, use the datetime formatter
        if has_time:
            return create_semantic_date_time_expression(date_str)

        # Get month name, day, and year
        month_name = parsed_date.strftime("%B")
        day = parsed_date.day
        year = parsed_date.year

        # Calculate week of month (approximate)
        week_of_month = (day - 1) // 7 + 1

        # Convert week number to ordinal word
        week_ordinals = ["first", "second", "third", "fourth", "fifth"]
        if 1 <= week_of_month <= 5:
            week_ordinal = week_ordinals[week_of_month - 1]
        else:
            week_ordinal = f"{week_of_month}th"  # Fallback if calculation is off

        # Calculate day of week
        day_of_week = parsed_date.strftime("%A")

        # Calculate quarter and convert to ordinal word
        quarter_num = (parsed_date.month - 1) // 3 + 1
        quarter_ordinals = {1: "first", 2: "second", 3: "third", 4: "fourth"}
        quarter_ordinal = quarter_ordinals.get(quarter_num, f"{quarter_num}th")

        # Calculate decade as ordinal within century
        decade_in_century = (year % 100) // 10 + 1

        # Convert decade to ordinal word
        decade_ordinals = {1: "first", 2: "second", 3: "third", 4: "fourth",
                           5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
                           9: "ninth", 10: "tenth"}
        decade_ordinal = decade_ordinals.get(decade_in_century, f"{decade_in_century}th")

        # Calculate century
        century = (year // 100) + 1

        # Format century as ordinal
        century_ordinals = {1: "1st", 2: "2nd", 3: "3rd"}
        century_ordinal = century_ordinals.get(century, f"{century}th")

        # Format as more descriptive natural language
        return f"the month of {month_name} ({quarter_ordinal} quarter), in the {week_ordinal} week, on {day_of_week} day {day}, in the year {year}, during the {decade_ordinal} decade of the {century_ordinal} century"

    except Exception as e:
        logger.warning(f"Error converting date to semantic expression: {str(e)}")
        return date_str  # Return original on any error


@ttl_cache(maxsize=128, ttl=3600)
def create_semantic_date_time_expression(dt_str):
    """
    Convert a datetime string into a rich semantic natural language expression
    that includes both date and time information.

    Args:
        dt_str: A string representing a datetime

    Returns:
        A natural language representation with rich semantic context
    """
    try:
        # Import dateutil parser
        from dateutil import parser

        # Parse the datetime string
        parsed_dt = parser.parse(dt_str)

        # Generate date part
        date_part = create_semantic_date_expression(dt_str)

        # Generate time part
        time_part = create_semantic_time_expression(parsed_dt)

        # Combine them
        return f"{date_part}, {time_part}"

    except Exception as e:
        logger.warning(f"Error converting datetime to semantic expression: {str(e)}")
        return dt_str  # Return original on error


def create_semantic_temporal_expression(input_string: str) -> str:
    """
    Create a semantic temporal expression based on the detected temporal type.

    Args:
        input_string: Input string to analyze and convert

    Returns:
        Semantic natural language representation of the temporal data
    """
    temporal_type = detect_temporal_type(input_string)

    if temporal_type == TemporalType.DATE:
        return create_semantic_date_expression(input_string)
    elif temporal_type == TemporalType.TIME:
        from dateutil import parser
        try:
            time_obj = parser.parse(input_string).time()
            return create_semantic_time_expression(time_obj)
        except Exception:
            return input_string
    elif temporal_type == TemporalType.DATETIME:
        return create_semantic_date_time_expression(input_string)
    elif temporal_type == TemporalType.TIME_RANGE:
        return create_semantic_time_range_expression(input_string)
    else:
        return input_string  # Not a temporal string

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

        # Add common business time range expressions
        business_terms = []

        # Check for common business hours
        if (start_time.hour == 9 and start_time.minute == 0 and
                end_time.hour == 17 and end_time.minute == 0):
            business_terms.extend(["nine to five", "9-5", "standard business hours", "regular office hours"])
        elif (start_time.hour == 8 and start_time.minute == 0 and
              end_time.hour == 17 and end_time.minute == 0):
            business_terms.extend(["eight to five", "8-5", "extended business hours"])
        elif (start_time.hour == 9 and start_time.minute == 0 and
              end_time.hour == 18 and end_time.minute == 0):
            business_terms.extend(["nine to six", "9-6", "extended office hours"])

        # Check for lunch hours
        if start_time.hour == 12 and end_time.hour == 13:
            business_terms.extend(["lunch hour", "lunch break", "midday break"])

        # Combine with semantic expressions
        result = f"from {start_semantic} until {end_semantic}"
        if business_terms:
            result += f", {', '.join(business_terms)}"

        return result

    except Exception as e:
        logger.warning(f"Error creating semantic time range expression: {str(e)}")
        return time_range_str  # Return original on error


@ttl_cache(maxsize=128, ttl=3600)
def create_semantic_time_expression(time_obj):
    """
    Convert a time object into a lightweight semantic expression with essential business terms.

    Args:
        time_obj: A datetime or time object containing time information

    Returns:
        A minimal expansion with key business time terms
    """
    try:
        # Extract hour and minute from either a datetime or time object
        if hasattr(time_obj, 'hour'):
            hour = time_obj.hour
        else:
            return str(time_obj)

        if hasattr(time_obj, 'minute'):
            minute = time_obj.minute
        else:
            minute = 0

        # Essential business terms only
        essential_terms = [f"{hour:02d}{minute:02d}"]

        # Military time (commonly used in business)

        # Common business time expressions
        if hour == 12 and minute == 0:
            essential_terms.extend(["noon", "midday"])
        elif hour == 0 and minute == 0:
            essential_terms.extend(["midnight"])
        elif minute == 30:
            hour_12 = hour % 12
            if hour_12 == 0:
                hour_12 = 12
            essential_terms.append(f"half past {hour_12}")

        # Time of day (useful for business context)
        if 9 <= hour < 17:
            essential_terms.append("business hours")
        elif 12 <= hour < 14:
            essential_terms.append("lunch time")

        # Start with original time, add essential terms
        result = str(time_obj)
        if essential_terms:
            result += f", {', '.join(essential_terms)}"

        return result

    except Exception as e:
        logger.warning(f"Error converting time to semantic expression: {str(e)}")
        return str(time_obj)


@ttl_cache(maxsize=128, ttl=3600)
def create_semantic_date_expression(date_str: str) -> str:
    """
    Convert a date string into a balanced semantic expression with practical business terms.

    Args:
        date_str: A string representing a date in various possible formats

    Returns:
        A balanced expansion with practical business terms for real-world search scenarios
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

        # Get basic date components
        month_name = parsed_date.strftime("%B")
        day = parsed_date.day
        year = parsed_date.year
        month_num = parsed_date.month
        day_of_week = parsed_date.strftime("%A")

        # Calculate quarter
        quarter_num = (parsed_date.month - 1) // 3 + 1
        quarter_ordinals = {1: "first", 2: "second", 3: "third", 4: "fourth"}
        quarter_ordinal = quarter_ordinals.get(quarter_num, f"{quarter_num}th")

        # Calculate week of month (practical for business)
        week_of_month = (day - 1) // 7 + 1
        week_ordinals = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
        week_ordinal = week_ordinals.get(week_of_month, f"{week_of_month}th")

        # Practical business terms
        practical_terms = []

        # Quarter terms (most important for business search)
        quarter_terms = {
            1: ["Q1", "first quarter"],
            2: ["Q2", "second quarter"],
            3: ["Q3", "third quarter"],
            4: ["Q4", "fourth quarter"]
        }
        practical_terms.extend(quarter_terms.get(quarter_num, []))

        # Month abbreviation and variations
        month_abbreviations = {
            1: ["Jan"], 2: ["Feb"], 3: ["Mar"], 4: ["Apr"], 5: ["May"], 6: ["Jun"],
            7: ["Jul"], 8: ["Aug"], 9: ["Sep"], 10: ["Oct"], 11: ["Nov"], 12: ["Dec"]
        }
        practical_terms.extend(month_abbreviations.get(month_num, []))

        # Seasonal terms (clear associations only)
        seasonal_terms = {
            1: ["winter"], 2: ["spring"], 3: ["summer"], 4: ["fall"]
        }
        if quarter_num in seasonal_terms:
            practical_terms.extend(seasonal_terms[quarter_num])

        # Week-based business terms (commonly searched)
        practical_terms.append(f"{week_ordinal} week")
        if week_of_month <= 2:
            practical_terms.append("early month")
        elif week_of_month >= 4:
            practical_terms.append("late month")
        else:
            practical_terms.append("mid month")

        # Month position terms (practical for business)
        if day <= 7:
            practical_terms.append("beginning of month")
        elif day >= 22:
            practical_terms.append("end of month")
            if quarter_num == 4 and month_num == 12:
                practical_terms.append("year end")
        elif 10 <= day <= 20:
            practical_terms.append("mid month")

        # Day of week (important for business day searches)
        practical_terms.append(day_of_week)
        if day_of_week in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            practical_terms.append("business day")
            practical_terms.append("weekday")
        else:
            practical_terms.append("weekend")

        # Business day of month (approximate - assumes ~22 business days per month)
        import calendar
        # Count business days from start of month to this date
        business_day_count = 0
        for d in range(1, day + 1):
            test_date = parsed_date.replace(day=d)
            if test_date.weekday() < 5:  # Monday=0, Sunday=6
                business_day_count += 1

        if business_day_count <= 22:  # Reasonable business day range
            if business_day_count <= 5:
                practical_terms.append("early business days")
            elif business_day_count >= 17:
                practical_terms.append("late business days")

            # Add specific business day ordinals for commonly referenced ones
            business_day_ordinals = {
                1: "first business day", 2: "second business day", 3: "third business day",
                5: "fifth business day", 10: "tenth business day", 15: "fifteenth business day",
                20: "twentieth business day"
            }
            if business_day_count in business_day_ordinals:
                practical_terms.append(business_day_ordinals[business_day_count])

        # Year and ISO format
        practical_terms.extend([str(year), f"{year}-{month_num:02d}"])

        # Fiscal year terms (practical ones only)
        if quarter_num == 4:
            practical_terms.extend(["fiscal year end", "year end"])
        elif quarter_num == 1:
            practical_terms.append("fiscal year start")

        # Start with original, add practical terms
        result = date_str
        if practical_terms:
            result += f", {', '.join(practical_terms)}"

        return result

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

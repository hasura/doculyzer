"""
Date extraction module using datefinder.

This module provides utilities for extracting dates from text and
converting them into a standardized format with enhanced temporal concepts
for storage with embeddings.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    import datefinder
except ImportError:
    raise ImportError("datefinder is required. Install with: pip install datefinder")

logger = logging.getLogger(__name__)


@dataclass
class ExtractedDate:
    """Standardized structure for an extracted date with enhanced temporal metadata."""

    # Core date information (always present)
    original_text: str
    datetime_obj: Optional[datetime]  # May be None for very vague dates like "1990s"
    iso_string: Optional[str] = None

    # Position in the source text
    start_position: int = -1
    end_position: int = -1
    context: str = ""

    # Basic date components (nullable when not specific enough)
    year: Optional[int] = None
    month: Optional[int] = None        # None for "Q2 2024", "Spring 2023", "1990s"
    day: Optional[int] = None          # None for "January 2024", "Q2 2024"
    hour: Optional[int] = None         # None when no time specified
    minute: Optional[int] = None       # None when no time specified
    second: Optional[int] = None       # None when no time specified
    timestamp: Optional[float] = None  # None when date is too vague

    # Extended temporal concepts (nullable when not applicable)
    decade: Optional[int] = None                    # 1990, 2000, 2010, etc.
    century: Optional[int] = None                   # 20, 21, etc.
    quarter: Optional[int] = None                   # 1, 2, 3, 4 (only when specific enough)
    season: Optional[str] = None                    # spring, summer, fall, winter
    day_of_week: Optional[str] = None              # monday, tuesday, etc. (only for specific dates)
    day_of_year: Optional[int] = None              # 1-366 (only for specific dates)
    week_of_year: Optional[int] = None             # 1-53 (only for specific dates)
    is_weekend: Optional[bool] = None              # Only when we have specific day
    is_holiday_season: Optional[bool] = None       # Only when we have month info

    # Business/Academic periods (nullable when not specific enough)
    fiscal_quarter: Optional[int] = None            # Q1, Q2, Q3, Q4 (only when specific enough)
    fiscal_year: Optional[int] = None              # Fiscal year (e.g., FY2024)
    academic_semester: Optional[str] = None        # fall, spring, summer (only when specific enough)
    academic_year: Optional[str] = None           # 2023-2024 (only when specific enough)

    # Time of day categories (nullable when no time specified)
    time_of_day: Optional[str] = None             # morning, afternoon, evening, night
    is_business_hours: Optional[bool] = None       # Only when we have specific date+time

    # Relative/contextual information
    is_relative: bool = False          # "yesterday", "next week"
    is_partial: bool = False          # missing components
    date_type: str = "absolute"       # absolute, relative, partial, vague
    relative_reference: str = ""      # what it's relative to
    specificity_level: str = "full"   # full, date_only, month_only, quarter_only, year_only, decade_only

    # Formatting and cultural context
    date_format_detected: str = ""    # MM/DD/YYYY, DD/MM/YYYY, etc.
    locale_hint: str = "US"          # US, EU, ISO, etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage, handling nullable fields properly."""
        return {
            # Core information
            'original_text': self.original_text,
            'iso_string': self.iso_string,
            'timestamp': self.timestamp,
            'start_position': self.start_position,
            'end_position': self.end_position,
            'context': self.context,

            # Basic components (nullable)
            'year': self.year,
            'month': self.month,
            'day': self.day,
            'hour': self.hour,
            'minute': self.minute,
            'second': self.second,

            # Extended temporal concepts (nullable)
            'decade': self.decade,
            'century': self.century,
            'quarter': self.quarter,
            'season': self.season,
            'day_of_week': self.day_of_week,
            'day_of_year': self.day_of_year,
            'week_of_year': self.week_of_year,
            'is_weekend': self.is_weekend,
            'is_holiday_season': self.is_holiday_season,

            # Business/Academic (nullable)
            'fiscal_quarter': self.fiscal_quarter,
            'fiscal_year': self.fiscal_year,
            'academic_semester': self.academic_semester,
            'academic_year': self.academic_year,

            # Time categories (nullable)
            'time_of_day': self.time_of_day,
            'is_business_hours': self.is_business_hours,

            # Context
            'is_relative': self.is_relative,
            'is_partial': self.is_partial,
            'date_type': self.date_type,
            'relative_reference': self.relative_reference,
            'specificity_level': self.specificity_level,

            # Format detection
            'date_format_detected': self.date_format_detected,
            'locale_hint': self.locale_hint
        }


class DateExtractor:
    """Enhanced date extractor using datefinder with comprehensive temporal analysis."""

    def __init__(self,
                 context_chars: int = 50,
                 min_year: int = 1900,
                 max_year: int = 2100,
                 fiscal_year_start_month: int = 10,  # October (US federal)
                 default_locale: str = "US"):
        """
        Initialize the date extractor.

        Args:
            context_chars: Number of characters to include around date for context
            min_year: Minimum valid year for extracted dates
            max_year: Maximum valid year for extracted dates
            fiscal_year_start_month: Month when fiscal year starts (1-12)
            default_locale: Default locale for date format detection
        """
        self.context_chars = context_chars
        self.min_year = min_year
        self.max_year = max_year
        self.fiscal_year_start_month = fiscal_year_start_month
        self.default_locale = default_locale

    def extract_dates(self, text: str) -> List[ExtractedDate]:
        """
        Extract all dates from the given text with comprehensive temporal analysis.

        Args:
            text: Text to extract dates from

        Returns:
            List of ExtractedDate objects
        """
        extracted_dates = []

        try:
            # First, extract specific dates using datefinder
            matches = datefinder.find_dates(text, source=True, strict=False)

            for date_obj, original_text in matches:
                if self._is_valid_date(date_obj):
                    # Find position of the original text in the full text
                    start_pos = text.find(original_text)
                    end_pos = start_pos + len(original_text) if start_pos >= 0 else -1

                    # Extract context around the date
                    context = self._extract_context(text, start_pos, end_pos) if start_pos >= 0 else ""

                    # Analyze the date comprehensively
                    extracted_date = self._analyze_date_comprehensively(
                        original_text, date_obj, start_pos, end_pos, context
                    )

                    extracted_dates.append(extracted_date)

            # Also look for vague temporal references that datefinder might miss
            vague_dates = self._extract_vague_temporal_references(text)
            extracted_dates.extend(vague_dates)

        except Exception as e:
            logger.warning(f"Error extracting dates from text: {e}")

        return extracted_dates

    def _analyze_date_comprehensively(self, original_text: str, date_obj: datetime,
                                     start_pos: int, end_pos: int, context: str) -> ExtractedDate:
        """Perform comprehensive analysis of a date to extract all temporal metadata."""

        # Determine specificity level based on what's available
        specificity = self._determine_specificity_level(original_text, date_obj)

        # Basic date components (only when specific enough)
        year = date_obj.year if specificity not in ["decade_only"] else None
        month = date_obj.month if specificity not in ["decade_only", "year_only"] else None
        day = date_obj.day if specificity in ["full", "date_only"] else None
        hour = date_obj.hour if specificity == "full" and date_obj.hour != 0 else None
        minute = date_obj.minute if specificity == "full" and date_obj.minute != 0 else None
        second = date_obj.second if specificity == "full" and date_obj.second != 0 else None

        # Extended temporal calculations (only when specific enough)
        decade = (date_obj.year // 10) * 10 if year else None
        century = (date_obj.year - 1) // 100 + 1 if year else None
        quarter = (date_obj.month - 1) // 3 + 1 if month else None
        season = self._get_season(date_obj.month) if month else None

        # Day-specific calculations (only for specific dates)
        day_of_week = date_obj.strftime('%A').lower() if day else None
        day_of_year = date_obj.timetuple().tm_yday if day else None
        week_of_year = date_obj.isocalendar()[1] if day else None
        is_weekend = date_obj.weekday() >= 5 if day else None
        is_holiday_season = month in [11, 12] if month else None

        # Business/Academic periods (only when specific enough)
        fiscal_quarter, fiscal_year = self._get_fiscal_period(date_obj) if month else (None, None)
        academic_semester = self._get_academic_semester(date_obj.month) if month else None
        academic_year = self._get_academic_year(date_obj) if month else None

        # Time of day analysis (only when time is specified)
        time_of_day = self._get_time_of_day(date_obj.hour) if hour is not None else None
        is_business_hours = self._is_business_hours(date_obj) if day and hour is not None else None

        # Relative/contextual analysis
        is_relative = self._is_relative_date(original_text)
        is_partial = self._is_partial_date(original_text, date_obj) or specificity != "full"
        date_type = self._classify_date_type(original_text, is_relative, is_partial)
        relative_reference = self._extract_relative_reference(original_text, context)

        # Format detection
        date_format_detected = self._detect_date_format(original_text)
        locale_hint = self._detect_locale_hint(original_text, date_format_detected)

        return ExtractedDate(
            # Core information
            original_text=original_text,
            datetime_obj=date_obj,
            iso_string=date_obj.isoformat(),
            start_position=start_pos,
            end_position=end_pos,
            context=context,

            # Basic components (nullable)
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
            timestamp=date_obj.timestamp(),

            # Extended temporal concepts (nullable)
            decade=decade,
            century=century,
            quarter=quarter,
            season=season,
            day_of_week=day_of_week,
            day_of_year=day_of_year,
            week_of_year=week_of_year,
            is_weekend=is_weekend,
            is_holiday_season=is_holiday_season,

            # Business/Academic (nullable)
            fiscal_quarter=fiscal_quarter,
            fiscal_year=fiscal_year,
            academic_semester=academic_semester,
            academic_year=academic_year,

            # Time categories (nullable)
            time_of_day=time_of_day,
            is_business_hours=is_business_hours,

            # Context
            is_relative=is_relative,
            is_partial=is_partial,
            date_type=date_type,
            relative_reference=relative_reference,
            specificity_level=specificity,

            # Format detection
            date_format_detected=date_format_detected,
            locale_hint=locale_hint
        )

    def _extract_vague_temporal_references(self, text: str) -> List[ExtractedDate]:
        """Extract vague temporal references like '1990s', 'Q2', 'spring', etc."""
        vague_dates = []

        # Decade references (1990s, 2000s, etc.)
        decade_pattern = r'\b(19|20)\d0s\b'
        for match in re.finditer(decade_pattern, text, re.IGNORECASE):
            decade_str = match.group()
            decade = int(decade_str[:-1])  # Remove 's'

            extracted_date = ExtractedDate(
                original_text=decade_str,
                datetime_obj=None,
                iso_string=None,
                start_position=match.start(),
                end_position=match.end(),
                context=self._extract_context(text, match.start(), match.end()),
                decade=decade,
                century=(decade - 1) // 100 + 1,
                date_type="vague",
                specificity_level="decade_only",
                is_partial=True
            )
            vague_dates.append(extracted_date)

        # Quarter references (Q1, Q2, Q3, Q4)
        quarter_pattern = r'\bQ([1-4])(?:\s+(19|20)\d{2})?\b'
        for match in re.finditer(quarter_pattern, text, re.IGNORECASE):
            quarter = int(match.group(1))
            year_str = match.group(2)
            year = int(year_str) if year_str else None

            extracted_date = ExtractedDate(
                original_text=match.group(),
                datetime_obj=None,
                iso_string=None,
                start_position=match.start(),
                end_position=match.end(),
                context=self._extract_context(text, match.start(), match.end()),
                year=year,
                quarter=quarter,
                date_type="partial",
                specificity_level="quarter_only" if not year else "quarter_year"
            )

            if year:
                extracted_date.decade = (year // 10) * 10
                extracted_date.century = (year - 1) // 100 + 1
                extracted_date.fiscal_quarter, extracted_date.fiscal_year = self._get_fiscal_period_from_quarter(quarter, year)

            vague_dates.append(extracted_date)

        # Season references (spring, summer, fall, winter)
        season_pattern = r'\b(spring|summer|fall|autumn|winter)(?:\s+(19|20)\d{2})?\b'
        for match in re.finditer(season_pattern, text, re.IGNORECASE):
            season = match.group(1).lower()
            if season == "autumn":
                season = "fall"
            year_str = match.group(2)
            year = int(year_str) if year_str else None

            extracted_date = ExtractedDate(
                original_text=match.group(),
                datetime_obj=None,
                iso_string=None,
                start_position=match.start(),
                end_position=match.end(),
                context=self._extract_context(text, match.start(), match.end()),
                year=year,
                season=season,
                date_type="partial",
                specificity_level="season_only" if not year else "season_year"
            )

            if year:
                extracted_date.decade = (year // 10) * 10
                extracted_date.century = (year - 1) // 100 + 1
                extracted_date.academic_semester = self._get_academic_semester_from_season(season)
                extracted_date.academic_year = self._get_academic_year_from_season_year(season, year)

            vague_dates.append(extracted_date)

        return vague_dates

    @staticmethod
    def _determine_specificity_level(original_text: str, date_obj: datetime) -> str:
        """Determine how specific the date is based on original text."""
        text_lower = original_text.lower()

        # Check for time indicators
        time_indicators = ['am', 'pm', ':', 'morning', 'afternoon', 'evening', 'night']
        has_time = any(indicator in text_lower for indicator in time_indicators)

        # Check for day indicators
        has_day = any(char.isdigit() for char in original_text if char not in ['-', '/', '.', ' '])
        day_indicators = ['st', 'nd', 'rd', 'th', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        has_day_ref = any(indicator in text_lower for indicator in day_indicators)

        # Check for month indicators
        month_names = ['january', 'february', 'march', 'april', 'may', 'june',
                      'july', 'august', 'september', 'october', 'november', 'december',
                      'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        has_month = any(month in text_lower for month in month_names)

        # Check for quarter/season indicators
        quarter_season_indicators = ['q1', 'q2', 'q3', 'q4', 'spring', 'summer', 'fall', 'autumn', 'winter']
        has_quarter_season = any(indicator in text_lower for indicator in quarter_season_indicators)

        if has_time and (has_day or has_day_ref):
            return "full"
        elif has_day or has_day_ref:
            return "date_only"
        elif has_month:
            return "month_only"
        elif has_quarter_season:
            return "quarter_only"
        elif any(str(year) in original_text for year in range(1900, 2100)):
            return "year_only"
        else:
            return "vague"

    @staticmethod
    def _get_season(month: int) -> str:
        """Determine the season based on month (Northern Hemisphere)."""
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "fall"
        return ""

    def _get_fiscal_period(self, date_obj: datetime) -> tuple[Optional[int], Optional[int]]:
        """Calculate fiscal quarter and year based on fiscal year start month."""
        month = date_obj.month
        year = date_obj.year

        # Adjust for fiscal year
        if month >= self.fiscal_year_start_month:
            fiscal_year = year + 1
            fiscal_month = month - self.fiscal_year_start_month + 1
        else:
            fiscal_year = year
            fiscal_month = month + (12 - self.fiscal_year_start_month) + 1

        fiscal_quarter = (fiscal_month - 1) // 3 + 1

        return fiscal_quarter, fiscal_year

    def _get_fiscal_period_from_quarter(self, quarter: int, year: int) -> tuple[Optional[int], Optional[int]]:
        """Calculate fiscal quarter and year for a given calendar quarter and year."""
        # This is a simplified calculation - in reality, fiscal quarters
        # might not align perfectly with calendar quarters
        if self.fiscal_year_start_month == 10:  # October start
            if quarter >= 1:  # Q1 (Jan-Mar) is actually Q2 of fiscal year
                fiscal_quarter = quarter + 1 if quarter < 4 else 1
                fiscal_year = year + 1 if quarter >= 1 else year
            else:
                fiscal_quarter = quarter
                fiscal_year = year
        else:
            # Default to calendar year alignment
            fiscal_quarter = quarter
            fiscal_year = year

        return fiscal_quarter, fiscal_year

    @staticmethod
    def _get_academic_semester(month: int) -> Optional[str]:
        """Determine academic semester based on month."""
        if month in [8, 9, 10, 11, 12]:
            return "fall"
        elif month in [1, 2, 3, 4, 5]:
            return "spring"
        elif month in [6, 7]:
            return "summer"
        return None

    @staticmethod
    def _get_academic_year(date_obj: datetime) -> Optional[str]:
        """Get academic year string (e.g., '2023-2024')."""
        month = date_obj.month
        year = date_obj.year

        if month >= 8:  # Fall semester starts academic year
            return f"{year}-{year + 1}"
        else:
            return f"{year - 1}-{year}"

    @staticmethod
    def _get_academic_semester_from_season(season: str) -> Optional[str]:
        """Get academic semester from season."""
        season_to_semester = {
            "fall": "fall",
            "winter": "spring",  # Winter often overlaps with spring semester
            "spring": "spring",
            "summer": "summer"
        }
        return season_to_semester.get(season)

    @staticmethod
    def _get_academic_year_from_season_year(season: str, year: int) -> Optional[str]:
        """Get academic year string from season and year."""
        if season in ["fall"]:
            return f"{year}-{year + 1}"
        elif season in ["spring", "winter"]:
            return f"{year - 1}-{year}"
        elif season == "summer":
            # Summer can be either end of previous year or beginning of next
            return f"{year - 1}-{year}"
        return None

    @staticmethod
    def _get_time_of_day(hour: int) -> str:
        """Categorize time of day."""
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    @staticmethod
    def _is_business_hours(date_obj: datetime) -> bool:
        """Check if datetime falls within standard business hours."""
        # Monday = 0, Sunday = 6
        is_weekday = date_obj.weekday() < 5
        is_business_time = 9 <= date_obj.hour < 17
        return is_weekday and is_business_time

    @staticmethod
    def _is_relative_date(text: str) -> bool:
        """Determine if a date expression is relative."""
        relative_indicators = [
            'yesterday', 'today', 'tomorrow', 'last', 'next', 'ago', 'from now',
            'this week', 'next week', 'last week', 'this month', 'next month', 'last month',
            'this year', 'next year', 'last year', 'this quarter', 'next quarter', 'last quarter'
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in relative_indicators)

    @staticmethod
    def _is_partial_date(text: str, date_obj: datetime) -> bool:
        """Determine if a date is partial (missing components)."""
        current_year = datetime.now().year
        return (date_obj.year == current_year and
                str(current_year) not in text and
                len(text.strip()) < 10)

    @staticmethod
    def _classify_date_type(text: str, is_relative: bool, is_partial: bool) -> str:
        """Classify the type of date."""
        if is_relative:
            return "relative"
        elif is_partial:
            return "partial"
        else:
            return "absolute"

    @staticmethod
    def _extract_relative_reference(original_text: str, context: str) -> str:
        """Extract what a relative date is relative to."""
        relative_indicators = [
            "last", "next", "this", "yesterday", "today", "tomorrow",
            "ago", "from now", "before", "after", "since", "until"
        ]

        text_lower = original_text.lower()
        context_lower = context.lower()

        found_indicators = []
        for indicator in relative_indicators:
            if indicator in text_lower or indicator in context_lower:
                found_indicators.append(indicator)

        return ", ".join(found_indicators) if found_indicators else ""

    @staticmethod
    def _detect_date_format(text: str) -> str:
        """Detect the likely date format from the original text."""
        # Common date format patterns
        if re.match(r'\d{1,2}/\d{1,2}/\d{4}', text):
            return "MM/DD/YYYY"
        elif re.match(r'\d{1,2}-\d{1,2}-\d{4}', text):
            return "MM-DD-YYYY"
        elif re.match(r'\d{4}-\d{1,2}-\d{1,2}', text):
            return "YYYY-MM-DD"
        elif re.match(r'\d{1,2}\.\d{1,2}\.\d{4}', text):
            return "MM.DD.YYYY"
        elif re.match(r'[A-Za-z]+\s+\d{1,2},?\s+\d{4}', text):
            return "Month DD, YYYY"
        elif re.match(r'\d{1,2}\s+[A-Za-z]+\s+\d{4}', text):
            return "DD Month YYYY"
        elif re.match(r'[A-Za-z]+\s+\d{4}', text):
            return "Month YYYY"
        return "unknown"

    def _detect_locale_hint(self, text: str, date_format: str) -> str:
        """Detect likely locale based on date format."""
        if date_format in ["YYYY-MM-DD"]:
            return "ISO"
        elif date_format in ["DD/MM/YYYY", "DD-MM-YYYY", "DD.MM.YYYY"]:
            return "EU"
        elif date_format in ["MM/DD/YYYY", "MM-DD-YYYY"]:
            return "US"
        return self.default_locale

    def extract_dates_as_dicts(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract dates and return as dictionaries (convenient for storage).

        Args:
            text: Text to extract dates from

        Returns:
            List of date dictionaries
        """
        dates = self.extract_dates(text)
        return [date.to_dict() for date in dates]

    def _is_valid_date(self, date_obj: datetime) -> bool:
        """Check if the date is within valid range."""
        return self.min_year <= date_obj.year <= self.max_year

    def _extract_context(self, text: str, start_pos: int, end_pos: int) -> str:
        """Extract context around a date."""
        if start_pos < 0 or end_pos < 0:
            return ""

        context_start = max(0, start_pos - self.context_chars)
        context_end = min(len(text), end_pos + self.context_chars)

        context = text[context_start:context_end]

        # Add ellipsis if we're not at the boundaries
        if context_start > 0:
            context = "..." + context
        if context_end < len(text):
            context = context + "..."

        return context.strip()


# Convenience functions for simple use cases
def extract_dates_from_text(text: str, **kwargs) -> List[ExtractedDate]:
    """
    Simple function to extract dates from text.

    Args:
        text: Text to extract dates from
        **kwargs: Additional arguments for DateExtractor

    Returns:
        List of ExtractedDate objects
    """
    extractor = DateExtractor(**kwargs)
    return extractor.extract_dates(text)


def extract_dates_as_dicts(text: str, **kwargs) -> List[Dict[str, Any]]:
    """
    Simple function to extract dates from text as dictionaries.

    Args:
        text: Text to extract dates from
        **kwargs: Additional arguments for DateExtractor

    Returns:
        List of date dictionaries
    """
    extractor = DateExtractor(**kwargs)
    return extractor.extract_dates_as_dicts(text)


# Demo/test function
def demo():
    """Demonstrate the enhanced date extraction functionality."""

    test_texts = [
        # Full specificity
        "The meeting is on January 15, 2024 at 2:30 PM.",

        # Date only
        "The report is due March 20, 2024.",

        # Month/Year only
        "The project started in January 2024.",

        # Quarter specificity
        "Q4 2023 earnings were strong.",
        "Our Q2 performance exceeded expectations.",

        # Season specificity
        "Summer 2023 was particularly busy.",
        "The fall semester begins soon.",

        # Year only
        "The company was founded in 1995.",

        # Decade only
        "Technology advanced rapidly in the 1990s.",
        "The 2000s brought many changes.",

        # Relative dates
        "The meeting was yesterday at 3 PM.",
        "Next Friday we have a deadline.",

        # Original demo text
        """
        Meeting Notes - Project Alpha

        The project kicked off on January 15, 2024, with the initial planning phase.
        We have scheduled the next review for March 20th at 2:00 PM.

        Key deadlines:
        - Design phase completion: Feb 28, 2024
        - Development start: Next Monday  
        - Beta testing: Last week of April
        - Final delivery: Q2 2024

        Please note that yesterday's meeting was cancelled and rescheduled for tomorrow.
        Also, we had a call on 05/15/2023 and another one scheduled for 12/25/2024.
        """
    ]

    print("=== Enhanced Date Extraction Demo ===")

    extractor = DateExtractor(context_chars=30)

    for i, text in enumerate(test_texts, 1):
        if len(text) > 100:  # Long text
            print(f"{i}. Long text sample ({len(text)} characters)")
            dates = extractor.extract_dates(text)
            print(f"   Found {len(dates)} dates:")
        else:  # Short text
            print(f"{i}. Text: '{text}'")
            dates = extractor.extract_dates(text)

        for j, date_info in enumerate(dates, 1):
            print(f"   Date {j}: '{date_info.original_text}'")
            print(f"     Specificity: {date_info.specificity_level}")
            print(f"     Type: {date_info.date_type}")

            # Show only non-null fields
            if date_info.datetime_obj:
                print(f"     Parsed: {date_info.datetime_obj}")
            if date_info.year:
                print(f"     Year: {date_info.year}")
            if date_info.month:
                print(f"     Month: {date_info.month}")
            if date_info.day:
                print(f"     Day: {date_info.day}")
            if date_info.hour is not None:
                print(f"     Time: {date_info.hour:02d}:{date_info.minute:02d}")

            if date_info.season:
                print(f"     Season: {date_info.season}")
            if date_info.quarter:
                print(f"     Quarter: Q{date_info.quarter}")
            if date_info.decade:
                print(f"     Decade: {date_info.decade}s")
            if date_info.fiscal_year and date_info.fiscal_quarter:
                print(f"     Fiscal: FY{date_info.fiscal_year} Q{date_info.fiscal_quarter}")

            if date_info.relative_reference:
                print(f"     Relative: {date_info.relative_reference}")

        print()


if __name__ == "__main__":
    demo()

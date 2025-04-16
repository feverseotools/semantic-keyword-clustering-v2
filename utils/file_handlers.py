# utils/file_handlers.py
"""
File handling utilities for the keyword clustering application.

This module provides functions for processing uploaded files,
exporting results, and handling various file formats.
"""

import os
import pandas as pd
import streamlit as st
from io import StringIO, BytesIO
from typing import Optional, Dict, Any, Union, List, Tuple

def process_uploaded_file(file: Any, format_type: str = "no_header") -> Optional[pd.DataFrame]:
    """
    Process an uploaded CSV file and return a dataframe.
    
    Args:
        file: The uploaded file object from st.file_uploader
        format_type: Format type ("no_header" or "with_header")
        
    Returns:
        Processed dataframe or None if processing fails
    """
    try:
        if format_type == "no_header":
            # Single column with no header
            df = pd.read_csv(file, header=None, names=["keyword"])
            st.success(f"✅ Loaded {len(df)} keywords (no header format)")
        else:
            # With header - handle different possible formats
            df = pd.read_csv(file, header=0)
            
            # Handle different possible header names for the keyword column
            keyword_column_found = False
            for possible_name in ["keyword", "keywords", "Keyword", "Keywords", "KEYWORD", "KEYWORDS"]:
                if possible_name in df.columns:
                    df.rename(columns={possible_name: "keyword"}, inplace=True)
                    keyword_column_found = True
                    break
            
            # If no known keyword column name was found, assume first column is keywords
            if not keyword_column_found:
                first_column = df.columns[0]
                df.rename(columns={first_column: "keyword"}, inplace=True)
                st.info(f"Using '{first_column}' as the keyword column")
            
            st.success(f"✅ Loaded {len(df)} keywords with {len(df.columns)} columns")
        
        # Basic validation
        if "keyword" not in df.columns:
            st.error("Could not identify keyword column in the CSV")
            return None
        
        # Check for empty values
        empty_count = df["keyword"].isnull().sum()
        if empty_count > 0:
            st.warning(f"Found {empty_count} empty keyword values. These will be ignored.")
            df = df.dropna(subset=["keyword"])
        
        # Check for duplicate keywords
        duplicate_count = len(df) - len(df["keyword"].unique())
        if duplicate_count > 0:
            st.warning(f"Found {duplicate_count} duplicate keywords. Keeping only unique values.")
            df = df.drop_duplicates(subset=["keyword"])
        
        # Standardize column types
        df["keyword"] = df["keyword"].astype(str)
        
        # Handle search volume if present
        for vol_col in ["search_volume", "searchvolume", "volume", "search volume", "Search Volume"]:
            if vol_col in df.columns:
                df.rename(columns={vol_col: "search_volume"}, inplace=True)
                # Convert to numeric, errors='coerce' will convert non-numeric to NaN
                df["search_volume"] = pd.to_numeric(df["search_volume"], errors="coerce").fillna(0).astype(int)
                break
        
        # Handle competition data if present
        for comp_col in ["competition", "Competition", "keyword_difficulty", "difficulty"]:
            if comp_col in df.columns:
                df.rename(columns={comp_col: "competition"}, inplace=True)
                # Normalize to 0-1 range if over 1
                if df["competition"].max() > 1:
                    df["competition"] = df["competition"] / 100
                break
        
        return df
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        
        # Try to provide more specific error messages
        if "No such file" in str(e):
            st.error("File not found. Please check that the file exists and is accessible.")
        elif "Expecting" in str(e) and "delimiter" in str(e):
            st.error("CSV parsing error. Check that your file is correctly formatted and uses commas as delimiters.")
        elif "UTF-8" in str(e):
            st.error("Encoding error. Try saving your CSV as UTF-8 before uploading.")
        
        return None

def export_results(df: pd.DataFrame, 
                 cluster_insights: Dict[int, Dict[str, Any]], 
                 format: str = "csv") -> Union[str, bytes]:
    """
    Export clustering results to the specified format.
    
    Args:
        df: Dataframe with clustering results
        cluster_insights: Dictionary of cluster insights
        format: Export format ("csv", "excel", "json")
        
    Returns:
        Exported data as string or bytes depending on format
    """
    # Create a copy to avoid modifying the original
    export_df = df.copy()
    
    # Add cluster name if not already present
    if "cluster_name" not in export_df.columns:
        export_df["cluster_name"] = export_df["cluster_id"].apply(
            lambda cid: cluster_insights.get(cid, {}).get("cluster_name", f"Cluster {cid}")
        )
    
    # Add primary intent if not already present
    if "primary_intent" not in export_df.columns:
        export_df["primary_intent"] = export_df["cluster_id"].apply(
            lambda cid: cluster_insights.get(cid, {}).get("primary_intent", "Unknown")
        )
    
    if format == "csv":
        return export_df.to_csv(index=False)
    
    elif format == "excel":
        output = BytesIO()
        
        try:
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                # Main data sheet
                export_df.to_excel(writer, sheet_name="Keywords", index=False)
                
                # Cluster summary sheet
                summary_data = []
                for cluster_id, insights in cluster_insights.items():
                    row = {
                        "cluster_id": cluster_id,
                        "cluster_name": insights.get("cluster_name", f"Cluster {cluster_id}"),
                        "keyword_count": insights.get("keyword_count", 0),
                        "primary_intent": insights.get("primary_intent", "Unknown"),
                        "journey_phase": insights.get("journey_phase", "Unknown"),
                    }
                    
                    # Add additional metrics if available
                    if "quality_score" in insights:
                        row["quality_score"] = insights["quality_score"]
                    
                    if "total_volume" in insights:
                        row["total_volume"] = insights["total_volume"]
                    
                    summary_data.append(row)
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="Cluster Summary", index=False)
                
                # Recommendations sheet
                recommendations = []
                for cluster_id, insights in cluster_insights.items():
                    cluster_name = insights.get("cluster_name", f"Cluster {cluster_id}")
                    intent = insights.get("primary_intent", "Unknown")
                    
                    # Add recommendations based on intent
                    content_formats = insights.get("suggested_content_formats", [])
                    recommendation = {
                        "cluster_id": cluster_id,
                        "cluster_name": cluster_name,
                        "primary_intent": intent,
                        "recommended_formats": ", ".join(content_formats[:5])
                    }
                    
                    recommendations.append(recommendation)
                
                recommendations_df = pd.DataFrame(recommendations)
                recommendations_df.to_excel(writer, sheet_name="Content Recommendations", index=False)
            
            return output.getvalue()
        
        except Exception as e:
            st.error(f"Error creating Excel file: {str(e)}")
            return export_df.to_csv(index=False)  # Fallback to CSV
    
    elif format == "json":
        # Create a more structured JSON format
        result = {
            "metadata": {
                "total_keywords": len(export_df),
                "total_clusters": len(cluster_insights),
                "generated_at": pd.Timestamp.now().isoformat()
            },
            "clusters": {},
            "keywords": export_df.to_dict(orient="records")
        }
        
        # Add cluster data
        for cluster_id, insights in cluster_insights.items():
            result["clusters"][str(cluster_id)] = insights
        
        # Convert to JSON string
        import json
        return json.dumps(result, indent=2)
    
    else:
        st.error(f"Unsupported export format: {format}")
        return export_df.to_csv(index=False)  # Fallback to CSV

def detect_file_format(file: Any) -> str:
    """
    Attempt to detect the format of a CSV file.
    
    Args:
        file: The uploaded file object
        
    Returns:
        Detected format ("no_header" or "with_header")
    """
    try:
        # Read the first few lines of the file
        file.seek(0)
        sample = file.read(1024).decode()
        file.seek(0)  # Reset file pointer
        
        # Split into lines
        lines = sample.strip().split("\n")
        
        if len(lines) < 2:
            return "no_header"  # Not enough lines to tell
        
        # Check if first line could be a header
        first_line = lines[0].strip()
        second_line = lines[1].strip()
        
        # Count commas in first two lines
        first_commas = first_line.count(",")
        second_commas = second_line.count(",")
        
        # If they have the same number of commas, check if first line contains
        # common header names
        if first_commas == second_commas:
            header_indicators = ["keyword", "search", "volume", "cpc", "competition", 
                                "difficulty", "month", "traffic", "score"]
            
            first_lower = first_line.lower()
            if any(indicator in first_lower for indicator in header_indicators):
                return "with_header"
            
            # Check if first row has non-numeric data while second has numeric
            first_cols = first_line.split(",")
            second_cols = second_line.split(",")
            
            if len(first_cols) > 1 and len(second_cols) > 1:
                has_numeric_second = any(col.strip().replace(".", "").isdigit() for col in second_cols[1:])
                has_alpha_first = any(not col.strip().replace(".", "").isdigit() for col in first_cols[1:])
                
                if has_numeric_second and has_alpha_first:
                    return "with_header"
        
        # If unsure, default to no header
        return "no_header"
        
    except Exception:
        return "no_header"  # Default to no header on error

def prepare_download_link(data: Union[str, bytes], filename: str, mimetype: str) -> None:
    """
    Prepare a download link for exporting data.
    
    Args:
        data: The data to export
        filename: Name of the download file
        mimetype: MIME type of the file
    """
    st.download_button(
        label=f"Download {filename.split('.')[-1].upper()}",
        data=data,
        file_name=filename,
        mime=mimetype,
        use_container_width=True
    )

def detect_and_read_sample(file: Any, max_rows: int = 5) -> Optional[pd.DataFrame]:
    """
    Detect file format and read a sample for preview.
    
    Args:
        file: The uploaded file object
        max_rows: Maximum number of rows to read
        
    Returns:
        Sample dataframe or None if reading fails
    """
    try:
        format_type = detect_file_format(file)
        
        if format_type == "no_header":
            return pd.read_csv(file, header=None, names=["keyword"], nrows=max_rows)
        else:
            return pd.read_csv(file, header=0, nrows=max_rows)
    
    except Exception as e:
        st.error(f"Error reading sample: {str(e)}")
        return None

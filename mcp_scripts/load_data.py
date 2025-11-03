"""
MCP tool to download GEO data.
"""
# pip install geoparse
from mcp.server.fastmcp import FastMCP
import GEOparse as GEO
import os

# Create an MCP server
mcp = FastMCP("load-data-mcp")

@mcp.tool()
def load_data(accession_id: str, download_dir: str) -> str:
    """
    Downloads GEO data.
    
    Args:
        accession_id: The GEO accession ID (e.g., 'GSE245274').
        download_dir: The directory to download the data to.
        
    Returns:
        A message indicating the download status.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    try:
        gse = GEO.get_GEO(geo=accession_id, destdir=download_dir)
        return f"Data for {accession_id} downloaded to {download_dir}"
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
   # Run as an MCP stdio server
   mcp.run(transport="stdio")
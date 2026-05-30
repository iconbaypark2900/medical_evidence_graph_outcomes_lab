#!/usr/bin/env python3
"""
Main entry point for the Medical Evidence Graph & Outcomes Insight Lab

This script provides a unified interface to start and manage the various services.
"""

import asyncio
import argparse
import json
import os
from pathlib import Path


def load_config(config_path: str = "config/settings.json") -> dict:
    """
    Load the configuration from the specified path
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        return json.load(f)


def start_service(service_name: str):
    """
    Start a specific service by name
    """
    # Import and start the appropriate service
    if service_name == "evidence_ingestion_service":
        from src.evidence_ingestion_service.main import main as ingestion_main
        return ingestion_main
    elif service_name == "evidence_graph_service":
        from src.evidence_graph_service.main import main as graph_main
        return graph_main
    elif service_name == "graph_rag_service":
        from src.graph_rag_service.main import main as rag_main
        return rag_main
    elif service_name == "outcomes_analytics_service":
        from src.outcomes_analytics_service.main import main as outcomes_main
        return outcomes_main
    elif service_name == "pathway_guideline_service":
        from src.pathway_guideline_service.main import main as pathway_main
        return pathway_main
    else:
        raise ValueError(f"Unknown service: {service_name}")


async def start_all_services():
    """
    Start all enabled services based on configuration
    """
    config = load_config()
    
    tasks = []
    for service_name, service_config in config.get("services", {}).items():
        if service_config.get("enabled", False):
            print(f"Starting {service_name}...")
            main_func = start_service(service_name)
            task = asyncio.create_task(main_func())
            tasks.append(task)
    
    # Wait for all services to complete
    if tasks:
        await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser(description="Medical Evidence Graph & Outcomes Insight Lab")
    parser.add_argument("service", nargs="?", help="Name of the service to start (or 'all' for all services)")
    parser.add_argument("--config", default="config/settings.json", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    if args.service == "all":
        print("Starting all enabled services...")
        asyncio.run(start_all_services())
    elif args.service:
        print(f"Starting {args.service}...")
        main_func = start_service(args.service)
        asyncio.run(main_func())
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
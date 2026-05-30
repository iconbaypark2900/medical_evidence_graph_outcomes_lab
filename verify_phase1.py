#!/usr/bin/env python3
"""
Final verification script for Phase 1 of Medical Evidence Graph & Outcomes Insight Lab
This script demonstrates the complete implementation of Phase 1.
"""

import asyncio
import sys
from datetime import datetime


def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)


def print_section(title):
    """Print a section header"""
    print(f"\n--- {title} ---")


async def verify_database_connections():
    """Verify database connection functionality"""
    print_section("1. DATABASE CONNECTIONS VERIFICATION")
    
    try:
        # Test mock database connections
        from src.mock_databases_test import test_mock_databases
        success = await test_mock_databases()
        print(f"   ✅ Database connection tests: {'PASSED' if success else 'FAILED'}")
        return success
    except Exception as e:
        print(f"   ❌ Database connection tests: FAILED - {e}")
        return False


async def verify_data_ingestion():
    """Verify data ingestion functionality"""
    print_section("2. DATA INGESTION VERIFICATION")
    
    try:
        from src.data_ingestion import ingest_medical_evidence
        
        # Test ingestion with a small query
        evidence = await ingest_medical_evidence(["diabetes"], max_per_source=1)
        print(f"   ✅ Data ingestion: SUCCESS - Ingested {len(evidence)} pieces of evidence")
        if evidence:
            print(f"      Sample title: {evidence[0].title[:60]}...")
        return len(evidence) > 0
    except Exception as e:
        print(f"   ❌ Data ingestion: FAILED - {e}")
        return False


async def verify_integration():
    """Verify integration between components"""
    print_section("3. INTEGRATION VERIFICATION")
    
    try:
        from src.mock_integration_test import mock_integration_test
        success = await mock_integration_test()
        print(f"   ✅ Integration test: {'PASSED' if success else 'FAILED'}")
        return success
    except Exception as e:
        print(f"   ❌ Integration test: FAILED - {e}")
        return False


async def verify_services():
    """Verify that all service structures are in place"""
    print_section("4. SERVICE STRUCTURE VERIFICATION")
    
    services = [
        "evidence_ingestion_service",
        "evidence_graph_service", 
        "graph_rag_service",
        "outcomes_analytics_service",
        "pathway_guideline_service"
    ]
    
    success_count = 0
    for service in services:
        try:
            # Try to import the main module for each service
            module_path = f"src.{service}.main"
            __import__(module_path)
            print(f"   ✅ {service}: Structure in place")
            success_count += 1
        except ImportError:
            print(f"   ❌ {service}: Missing or invalid")
    
    all_services_ok = success_count == len(services)
    print(f"   Overall: {'PASSED' if all_services_ok else 'PARTIAL'} - {success_count}/{len(services)} services ready")
    return all_services_ok


async def main():
    """Main verification function"""
    print_header("MEDICAL EVIDENCE GRAPH & OUTCOMES INSIGHT LAB")
    print_header("PHASE 1 IMPLEMENTATION VERIFICATION")
    print(f"\nVerification Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all verification tests
    db_success = await verify_database_connections()
    ingestion_success = await verify_data_ingestion()
    integration_success = await verify_integration()
    services_success = await verify_services()
    
    # Summary
    print_section("5. VERIFICATION SUMMARY")
    
    results = {
        "Database Connections": db_success,
        "Data Ingestion": ingestion_success,
        "Integration": integration_success,
        "Service Structure": services_success
    }
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    for test, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test}: {status}")
    
    print(f"\nOverall Result: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 PHASE 1 IMPLEMENTATION SUCCESSFUL!")
        print("✅ All core components have been implemented and verified")
        print("✅ Ready to proceed with Phase 2: Real Database Deployment")
        print("✅ Foundation is solid for advanced analytics implementation")
        return True
    else:
        print(f"\n❌ PHASE 1 IMPLEMENTATION INCOMPLETE")
        print(f"❌ {total_count - passed_count} out of {total_count} tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
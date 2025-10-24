#!/usr/bin/env python3
"""
Test script for the query relevance checking feature
"""

import requests
import json
from typing import Dict, Any

# API Base URL
API_URL = "http://localhost:8000"

def test_relevance_check(query: str) -> Dict[str, Any]:
    """Test the /check-relevance endpoint"""
    print(f"\n{'='*70}")
    print(f"Testing Query: {query}")
    print(f"{'='*70}")
    
    try:
        response = requests.post(
            f"{API_URL}/check-relevance",
            json={"query": query},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Status: {response.status_code}")
            print(f"  Is Relevant: {result['is_relevant']}")
            print(f"  Reason: {result['reason']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            return result
        else:
            print(f"✗ Status: {response.status_code}")
            print(f"  Error: {response.text}")
            return None
    
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to API at {API_URL}")
        print("  Make sure the FastAPI server is running: python enhanced_fastapi_server.py")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def main():
    """Run test suite"""
    print("\n" + "="*70)
    print("Query Relevance Checking Test Suite")
    print("="*70)
    
    # Test cases: (query, expected_relevance)
    test_cases = [
        # Relevant queries
        ("What is portfolio optimization in finance?", True),
        ("How to use MATLAB for financial analysis?", True),
        ("Explain stock market volatility", True),
        ("What are derivative instruments?", True),
        ("How to perform Monte Carlo simulation in MATLAB?", True),
        ("What is the Sharpe ratio?", True),
        ("Explain time series analysis in economics", True),
        
        # Irrelevant queries
        ("How to cook pasta?", False),
        ("What's the best pizza recipe?", False),
        ("Tell me about sports teams", False),
        ("How to train a dog?", False),
        ("What's the weather today?", False),
        ("Tell me a joke", False),
        ("How to knit a sweater?", False),
    ]
    
    print(f"\nRunning {len(test_cases)} test cases...\n")
    
    results_summary = {
        "relevant_correct": 0,
        "relevant_wrong": 0,
        "irrelevant_correct": 0,
        "irrelevant_wrong": 0,
    }
    
    for query, expected_relevant in test_cases:
        result = test_relevance_check(query)
        
        if result:
            is_relevant = result['is_relevant']
            
            if is_relevant == expected_relevant:
                status = "✓ PASS"
                if expected_relevant:
                    results_summary["relevant_correct"] += 1
                else:
                    results_summary["irrelevant_correct"] += 1
            else:
                status = "✗ FAIL"
                if expected_relevant:
                    results_summary["relevant_wrong"] += 1
                else:
                    results_summary["irrelevant_wrong"] += 1
            
            print(f"  {status}")
        else:
            print("  ✗ ERROR - Could not connect to API")
    
    # Summary
    print(f"\n{'='*70}")
    print("Test Summary")
    print(f"{'='*70}")
    print(f"Relevant queries correctly identified: {results_summary['relevant_correct']}")
    print(f"Relevant queries incorrectly rejected: {results_summary['relevant_wrong']}")
    print(f"Irrelevant queries correctly rejected: {results_summary['irrelevant_correct']}")
    print(f"Irrelevant queries incorrectly accepted: {results_summary['irrelevant_wrong']}")
    
    total_correct = results_summary["relevant_correct"] + results_summary["irrelevant_correct"]
    total_tests = len(test_cases)
    accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\nOverall Accuracy: {accuracy:.1f}% ({total_correct}/{total_tests})")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

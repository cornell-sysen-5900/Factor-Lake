"""
Test runner script for Factor-Lake project
Provides different test running options and configurations
"""
import sys
import subprocess


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running ALL tests...")
    print("=" * 60)
    return subprocess.call(['pytest', 'UnitTests/', '-v'])


def run_fast_tests():
    """Run only fast tests (exclude integration and slow tests)"""
    print("=" * 60)
    print("Running FAST tests only...")
    print("=" * 60)
    return subprocess.call(['pytest', 'UnitTests/', '-v', '-m', 'not slow'])


def run_unit_tests():
    """Run only unit tests"""
    print("=" * 60)
    print("Running UNIT tests...")
    print("=" * 60)
    return subprocess.call(['pytest', 'UnitTests/', '-v', '-k', 'not integration'])


def run_integration_tests():
    """Run only integration tests"""
    print("=" * 60)
    print("Running INTEGRATION tests...")
    print("=" * 60)
    return subprocess.call(['pytest', 'UnitTests/', '-v', '-k', 'integration'])


def run_specific_module(module_name):
    """Run tests for a specific module"""
    print("=" * 60)
    print(f"Running tests for {module_name}...")
    print("=" * 60)
    return subprocess.call(['pytest', f'UnitTests/test_{module_name}.py', '-v'])


def run_with_coverage():
    """Run tests with coverage report"""
    print("=" * 60)
    print("Running tests with coverage...")
    print("=" * 60)
    result = subprocess.call([
        'pytest', 'UnitTests/', '-v',
        '--cov=.', '--cov-report=html', '--cov-report=term'
    ])
    print("\nCoverage report generated in htmlcov/index.html")
    return result


def main():
    """Main test runner"""
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py [option]")
        print("\nOptions:")
        print("  all              - Run all tests")
        print("  fast             - Run fast tests only")
        print("  unit             - Run unit tests only")
        print("  integration      - Run integration tests only")
        print("  coverage         - Run with coverage report")
        print("  market           - Run market_object tests")
        print("  factors          - Run factor tests")
        print("  portfolio        - Run portfolio tests")
        print("  holdings         - Run calculate_holdings tests")
        print("  supabase         - Run Supabase integration tests")
        sys.exit(1)
    
    option = sys.argv[1].lower()
    
    if option == 'all':
        sys.exit(run_all_tests())
    elif option == 'fast':
        sys.exit(run_fast_tests())
    elif option == 'unit':
        sys.exit(run_unit_tests())
    elif option == 'integration':
        sys.exit(run_integration_tests())
    elif option == 'coverage':
        sys.exit(run_with_coverage())
    elif option in ['market', 'factors', 'portfolio', 'holdings', 'supabase']:
        module_map = {
            'market': 'market_object',
            'factors': 'factors',
            'portfolio': 'portfolio',
            'holdings': 'calculate_holdings',
            'supabase': 'supabase_integration'
        }
        sys.exit(run_specific_module(module_map[option]))
    else:
        print(f"Unknown option: {option}")
        sys.exit(1)


if __name__ == '__main__':
    main()
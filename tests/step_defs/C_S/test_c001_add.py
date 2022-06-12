from pytest_bdd import scenario, scenarios, given, when, then
import src.NOW.CodeSignal.c001_add as add
# Scenario: Add two integers
# Given integers a and b
# When a and b are added
# Then the result is a+b

scenarios('../../features/C_S/c001_add.feature')

#    scenario('../features/C_S/c001_add.feature', 'Add two integers')
# def test_add():
#     """
#     NOTE: Relative paths here are critical, 
#     depth of folder to be reflected in ../../ etc
#     """
#     pass

@given('integers a and b',  target_fixture='given_integers')
def given_integers():
    return [1, 3]
    # a = 1, b = 3


@when('a and b are added', target_fixture='when_added')
def when_added(given_integers):
    return given_integers[0] + given_integers[1]


@then('the result is a+b')
def then_result(given_integers, when_added):
    assert when_added == add.solution(given_integers[0], given_integers[1])

"""
Feature: CodeSignal c002 century from year
    1st challenge

    Scenario: Return the century from the year input
        Given a 4 digit integer year
        When a calculation is made
        Then the expected 2 digit integer century is returned
"""
from pytest_bdd import scenarios, given, when, then
import src.NOW.CodeSignal.c002_centuryfromyear as centuryfromyear


scenarios('../../features/C_S/c002_centuryfromyear.feature')
global INPUTcentury
INPUTcentury = 1

@given('a the "<inputyear>" as input')
def giventheinputyear(inputyear):
    return recordfortest(inputyear)

@when('a calculation is made')
def when_thecalculationismade():
    pass


@then('the expected 2 century is "<centuryexpected>"')
def compare_results(centuryexpected):
    assert centuryfromyear.solution(recordfortest.century) == centuryexpected


class recordfortest():
    def __init__(self, inputprovided):
        self.century = inputprovided

from src.NOW.CodeSignal import c001_add

class TestOur2():
    
    def test_2(self):
        a = 5
        b = 3
        assert a + b == c001_add.solution(a,b)
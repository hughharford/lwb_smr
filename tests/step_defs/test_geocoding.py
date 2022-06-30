from lwb_smr.Geocoding import Geocoding

def test_geocoding():
    geocode = Geocoding()
    res = geocode.get_geocoding('SW1A 1AA')

    assert len(res) == 5



if __name__ == '__main__':
    test_geocoding()

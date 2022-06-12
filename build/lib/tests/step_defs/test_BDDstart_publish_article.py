from pytest_bdd import scenarios, given, when, then

from src.BDDstart import BDDstart_publish_article

        
scenarios('../features/BDDstart_publish_article.feature')
def test_publish():
    pass #assert 11 == 1

@given("JimmyJimmy")
def author_user():
    pass

@given("I have an article", target_fixture="article")
def article():
    return BDDstart_publish_article.PublishArticle('JimmyJimmy')

@when("I update the new article with just a name")
def publish_untitled(article):
    article.setarticledetails(articlename='newbookname')

@when("I publish with name and author information")
def publish_article_with_details():
    BDDstart_publish_article.PublishArticle('bdd named author', 'how to run BDD article')

@then("I can see there are two articles published")
def get_number_published():
    numberArticles = BDDstart_publish_article.PublishArticle.showarticlelist()
    assert numberArticles == 2
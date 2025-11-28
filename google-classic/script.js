// Google Classic - JavaScript
// Simulates early 2000s Google search experience

// Focus on search box when page loads
window.onload = function() {
    var searchBox = document.getElementsByName('q')[0];
    if (searchBox) {
        searchBox.focus();
    }
};

// Handle "I'm Feeling Lucky" - redirects to first result
function feelingLucky(query) {
    // In classic Google, this would redirect to the first search result
    // For this simulation, we'll just go to search results
    return true;
}

// Store search history in a simple way (no localStorage in early 2000s, using cookies simulation)
function addToHistory(query) {
    // Simplified - early Google didn't really do this client-side
    var history = getSearchHistory();
    if (history.indexOf(query) === -1) {
        history.unshift(query);
        if (history.length > 10) {
            history.pop();
        }
    }
}

function getSearchHistory() {
    // Simple in-memory storage for demo
    if (!window.searchHistory) {
        window.searchHistory = [];
    }
    return window.searchHistory;
}

// Simple form validation
function validateSearch(form) {
    var query = form.q.value;
    if (query.trim() === '') {
        return false;
    }
    return true;
}

// Handle radio button changes for search type
function setSearchType(type) {
    var radios = document.getElementsByName('meta');
    for (var i = 0; i < radios.length; i++) {
        if (radios[i].value === type) {
            radios[i].checked = true;
            break;
        }
    }
}

// Classic Google easter egg - searching for "google" shows special message
function checkEasterEggs(query) {
    var lowerQuery = query.toLowerCase();
    if (lowerQuery === 'google') {
        // In early Google, searching for "google" was common
        return true;
    }
    if (lowerQuery === 'elgoog') {
        // Mirror google easter egg
        return true;
    }
    return false;
}

// Generate mock search results for demonstration
function generateMockResults(query) {
    // This simulates what search results looked like in early 2000s
    var results = [];

    // Common early 2000s websites that would appear in results
    var mockSites = [
        {
            title: query + ' - Wikipedia, the free encyclopedia',
            url: 'http://www.wikipedia.org/wiki/' + encodeURIComponent(query),
            snippet: 'From Wikipedia, the free encyclopedia. ' + query + ' is a topic that has been documented extensively...'
        },
        {
            title: query + ' Information and Resources',
            url: 'http://www.about.com/' + encodeURIComponent(query),
            snippet: 'Learn more about ' + query + '. Find articles, guides, and expert advice on this topic.'
        },
        {
            title: 'Yahoo! Directory - ' + query,
            url: 'http://dir.yahoo.com/search?p=' + encodeURIComponent(query),
            snippet: 'Yahoo! Directory listing for ' + query + '. Browse categories and find related websites.'
        },
        {
            title: query + ' at Amazon.com',
            url: 'http://www.amazon.com/s?keywords=' + encodeURIComponent(query),
            snippet: 'Shop for ' + query + ' at Amazon.com. Free shipping on orders over $25.'
        },
        {
            title: query + ' - Encyclopædia Britannica',
            url: 'http://www.britannica.com/search?query=' + encodeURIComponent(query),
            snippet: 'Encyclopædia Britannica article on ' + query + '. Authoritative reference content.'
        },
        {
            title: 'CNET.com - ' + query + ' Reviews',
            url: 'http://www.cnet.com/search/?query=' + encodeURIComponent(query),
            snippet: 'Read reviews and get the latest news about ' + query + ' from CNET, the technology experts.'
        },
        {
            title: query + ' - HowStuffWorks',
            url: 'http://www.howstuffworks.com/search.php?terms=' + encodeURIComponent(query),
            snippet: 'Learn how ' + query + ' works. In-depth explanations and diagrams to help you understand.'
        },
        {
            title: query + ' Forums - pair Networks',
            url: 'http://forums.pair.com/search/' + encodeURIComponent(query),
            snippet: 'Discussion forums about ' + query + '. Join the community and share your knowledge.'
        },
        {
            title: query + ' - GeoCities',
            url: 'http://www.geocities.com/search?q=' + encodeURIComponent(query),
            snippet: 'GeoCities member pages about ' + query + '. Personal homepages and fan sites.'
        },
        {
            title: query + ' at eBay',
            url: 'http://search.ebay.com/' + encodeURIComponent(query),
            snippet: 'Find great deals on ' + query + ' at eBay. Auctions ending soon!'
        }
    ];

    return mockSites;
}

// Format number with commas (classic style)
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

// Get random search time (classic Google showed this)
function getSearchTime() {
    return (Math.random() * 0.5 + 0.1).toFixed(2);
}

// Classic Google did not have instant search - this mimics that behavior
// No autocomplete, no suggestions while typing

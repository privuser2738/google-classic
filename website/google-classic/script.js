// Google Classic - JavaScript
// Simulates early 2000s Google search experience

// Focus on search box when page loads
window.onload = function() {
    var searchBox = document.getElementsByName('q')[0];
    if (searchBox) {
        searchBox.focus();
    }
};

// Simple seeded random number generator for consistent results per query/page
function seededRandom(seed) {
    var x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
}

// Hash a string to a number for seeding
function hashString(str) {
    var hash = 0;
    for (var i = 0; i < str.length; i++) {
        var char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash;
    }
    return Math.abs(hash);
}

// Early 2000s website templates - these were the big sites back then
var siteTemplates = [
    // Major portals and directories
    { domain: 'www.yahoo.com', name: 'Yahoo!', type: 'directory' },
    { domain: 'www.dmoz.org', name: 'DMOZ Open Directory', type: 'directory' },
    { domain: 'dir.yahoo.com', name: 'Yahoo! Directory', type: 'directory' },
    { domain: 'www.looksmart.com', name: 'LookSmart', type: 'directory' },

    // Reference sites
    { domain: 'www.wikipedia.org', name: 'Wikipedia', type: 'encyclopedia' },
    { domain: 'www.britannica.com', name: 'Encyclopædia Britannica', type: 'encyclopedia' },
    { domain: 'www.howstuffworks.com', name: 'HowStuffWorks', type: 'reference' },
    { domain: 'www.about.com', name: 'About.com', type: 'reference' },
    { domain: 'www.answers.com', name: 'Answers.com', type: 'reference' },
    { domain: 'www.infoplease.com', name: 'Infoplease', type: 'reference' },

    // Shopping
    { domain: 'www.amazon.com', name: 'Amazon.com', type: 'shopping' },
    { domain: 'www.ebay.com', name: 'eBay', type: 'auction' },
    { domain: 'www.buy.com', name: 'Buy.com', type: 'shopping' },
    { domain: 'www.shopping.com', name: 'Shopping.com', type: 'shopping' },
    { domain: 'www.pricegrabber.com', name: 'PriceGrabber', type: 'shopping' },
    { domain: 'www.bizrate.com', name: 'BizRate', type: 'shopping' },

    // Tech sites
    { domain: 'www.cnet.com', name: 'CNET', type: 'tech' },
    { domain: 'www.zdnet.com', name: 'ZDNet', type: 'tech' },
    { domain: 'www.pcworld.com', name: 'PC World', type: 'tech' },
    { domain: 'www.pcmag.com', name: 'PC Magazine', type: 'tech' },
    { domain: 'www.wired.com', name: 'Wired News', type: 'tech' },
    { domain: 'slashdot.org', name: 'Slashdot', type: 'tech' },
    { domain: 'www.tomshardware.com', name: "Tom's Hardware", type: 'tech' },
    { domain: 'www.anandtech.com', name: 'AnandTech', type: 'tech' },
    { domain: 'www.techrepublic.com', name: 'TechRepublic', type: 'tech' },

    // News
    { domain: 'www.cnn.com', name: 'CNN.com', type: 'news' },
    { domain: 'www.msnbc.com', name: 'MSNBC', type: 'news' },
    { domain: 'news.bbc.co.uk', name: 'BBC News', type: 'news' },
    { domain: 'www.nytimes.com', name: 'The New York Times', type: 'news' },
    { domain: 'www.washingtonpost.com', name: 'Washington Post', type: 'news' },
    { domain: 'www.usatoday.com', name: 'USA Today', type: 'news' },

    // Forums and communities
    { domain: 'www.geocities.com', name: 'GeoCities', type: 'personal' },
    { domain: 'www.angelfire.com', name: 'Angelfire', type: 'personal' },
    { domain: 'www.tripod.com', name: 'Tripod', type: 'personal' },
    { domain: 'members.aol.com', name: 'AOL Members', type: 'personal' },
    { domain: 'www.xanga.com', name: 'Xanga', type: 'blog' },
    { domain: 'www.livejournal.com', name: 'LiveJournal', type: 'blog' },
    { domain: 'www.blogger.com', name: 'Blogger', type: 'blog' },

    // Educational
    { domain: 'www.edu', name: 'University Website', type: 'edu' },
    { domain: 'www.mit.edu', name: 'MIT', type: 'edu' },
    { domain: 'www.stanford.edu', name: 'Stanford University', type: 'edu' },
    { domain: 'www.berkeley.edu', name: 'UC Berkeley', type: 'edu' },

    // Software/Downloads
    { domain: 'www.download.com', name: 'Download.com', type: 'downloads' },
    { domain: 'www.tucows.com', name: 'Tucows', type: 'downloads' },
    { domain: 'www.sourceforge.net', name: 'SourceForge', type: 'downloads' },
    { domain: 'www.versiontracker.com', name: 'VersionTracker', type: 'downloads' },

    // Entertainment
    { domain: 'www.imdb.com', name: 'IMDb', type: 'entertainment' },
    { domain: 'www.rottentomatoes.com', name: 'Rotten Tomatoes', type: 'entertainment' },
    { domain: 'www.allmusic.com', name: 'All Music Guide', type: 'entertainment' },
    { domain: 'www.ign.com', name: 'IGN', type: 'gaming' },
    { domain: 'www.gamespot.com', name: 'GameSpot', type: 'gaming' },
    { domain: 'www.gamefaqs.com', name: 'GameFAQs', type: 'gaming' },

    // Health
    { domain: 'www.webmd.com', name: 'WebMD', type: 'health' },
    { domain: 'www.mayoclinic.com', name: 'Mayo Clinic', type: 'health' },
    { domain: 'www.healthcentral.com', name: 'HealthCentral', type: 'health' },

    // Travel
    { domain: 'www.expedia.com', name: 'Expedia', type: 'travel' },
    { domain: 'www.travelocity.com', name: 'Travelocity', type: 'travel' },
    { domain: 'www.orbitz.com', name: 'Orbitz', type: 'travel' },
    { domain: 'www.lonelyplanet.com', name: 'Lonely Planet', type: 'travel' },

    // Finance
    { domain: 'finance.yahoo.com', name: 'Yahoo! Finance', type: 'finance' },
    { domain: 'www.fool.com', name: 'The Motley Fool', type: 'finance' },
    { domain: 'www.marketwatch.com', name: 'MarketWatch', type: 'finance' }
];

// Snippet templates based on site type
var snippetTemplates = {
    directory: [
        'Browse our directory listing for {query}. Find websites organized by category.',
        'Directory results for {query}. Explore hand-picked websites and resources.',
        '{query} - Directory listing with subcategories and related sites.'
    ],
    encyclopedia: [
        '{query} - From the encyclopedia. {query} refers to a subject of significant interest...',
        'Encyclopedia article on {query}. Learn about the history, significance, and details.',
        '{query}. This article covers the main aspects and provides comprehensive information...'
    ],
    reference: [
        'Learn about {query}. Comprehensive guide with explanations and examples.',
        '{query} explained - Find out everything you need to know about this topic.',
        'Reference guide for {query}. Detailed information and expert resources.'
    ],
    shopping: [
        'Shop for {query}. Compare prices from top retailers. Free shipping available.',
        'Find great deals on {query}. Save money with our price comparison.',
        '{query} - Shop now and save. Thousands of products available.'
    ],
    auction: [
        'Buy {query} on auction. Bid now - auctions ending soon!',
        '{query} for sale. Find new and used items. Place your bid today.',
        'Auction listings for {query}. Great deals from sellers worldwide.'
    ],
    tech: [
        '{query} reviews, news, and downloads. Expert technology coverage.',
        'Technology news: {query}. Read reviews and get buying advice.',
        '{query} - Tech reviews, specs, and comparisons from industry experts.'
    ],
    news: [
        'Latest news on {query}. Breaking stories and in-depth coverage.',
        '{query} - News, analysis, and opinion from around the world.',
        'Read the latest {query} news and updates from trusted journalists.'
    ],
    personal: [
        '{query} - Personal homepage. Fan site with information and pictures.',
        'My {query} Page - Welcome to my site about {query}!',
        '{query} fansite. Created by a fan, for fans. Last updated 2003.'
    ],
    blog: [
        '{query} - Blog posts and personal thoughts on this topic.',
        'Blogging about {query}. Read my latest entries and leave a comment.',
        '{query} blog. Daily updates and musings from a dedicated writer.'
    ],
    edu: [
        '{query} - Academic resources and research from university archives.',
        'University course materials on {query}. Educational resources.',
        'Academic paper: {query}. Research and scholarly articles available.'
    ],
    downloads: [
        'Download {query} software. Free and shareware programs available.',
        '{query} - Free download. Latest version with reviews and screenshots.',
        'Software downloads for {query}. Freeware, shareware, and demos.'
    ],
    entertainment: [
        '{query} - Entertainment database with ratings and reviews.',
        'Find information about {query}. Cast, crew, reviews, and more.',
        '{query} guide. Comprehensive entertainment database and community.'
    ],
    gaming: [
        '{query} - Game reviews, cheats, codes, and walkthroughs.',
        'Gaming guide for {query}. FAQs, hints, and strategy guides.',
        '{query} cheats and codes. Complete game guide with tips.'
    ],
    health: [
        '{query} - Health information and medical resources.',
        'Medical information about {query}. Symptoms, treatments, and advice.',
        '{query} health guide. Expert medical information you can trust.'
    ],
    travel: [
        '{query} travel guide. Hotels, flights, and vacation packages.',
        'Plan your trip: {query}. Find deals on hotels and airfare.',
        '{query} - Travel information, reviews, and booking services.'
    ],
    finance: [
        '{query} - Financial news, stock quotes, and market analysis.',
        'Investing in {query}. Market data and financial research.',
        '{query} financial information. Quotes, charts, and expert analysis.'
    ]
};

// Generate varied results based on query and page number
function generateMockResults(query, pageNum) {
    if (!pageNum) pageNum = 1;

    var results = [];
    var seed = hashString(query + pageNum);
    var numResults = 10;

    // Shuffle the site templates based on seed
    var shuffledSites = siteTemplates.slice();
    for (var i = shuffledSites.length - 1; i > 0; i--) {
        var j = Math.floor(seededRandom(seed + i) * (i + 1));
        var temp = shuffledSites[i];
        shuffledSites[i] = shuffledSites[j];
        shuffledSites[j] = temp;
    }

    // Start index based on page (so we get different sites per page)
    var startIdx = ((pageNum - 1) * 10) % shuffledSites.length;

    for (var i = 0; i < numResults; i++) {
        var siteIdx = (startIdx + i) % shuffledSites.length;
        var site = shuffledSites[siteIdx];
        var snippets = snippetTemplates[site.type] || snippetTemplates.reference;
        var snippetIdx = Math.floor(seededRandom(seed + i + 100) * snippets.length);
        var snippet = snippets[snippetIdx].replace(/\{query\}/g, query);

        // Generate URL path
        var urlPaths = [
            '/search?q=' + encodeURIComponent(query),
            '/' + encodeURIComponent(query.toLowerCase().replace(/\s+/g, '_')),
            '/articles/' + encodeURIComponent(query.toLowerCase().replace(/\s+/g, '-')),
            '/wiki/' + encodeURIComponent(query.replace(/\s+/g, '_')),
            '/topic/' + encodeURIComponent(query.toLowerCase().replace(/\s+/g, '+')),
            '/info/' + encodeURIComponent(query.toLowerCase()),
            '/guide/' + encodeURIComponent(query.toLowerCase().replace(/\s+/g, '_'))
        ];
        var pathIdx = Math.floor(seededRandom(seed + i + 200) * urlPaths.length);

        // Generate title variations
        var titles = [
            query + ' - ' + site.name,
            site.name + ': ' + query,
            query + ' | ' + site.name,
            query + ' Information - ' + site.name,
            site.name + ' - ' + query + ' Guide',
            'Learn About ' + query + ' - ' + site.name,
            query + ' Resources at ' + site.name
        ];
        var titleIdx = Math.floor(seededRandom(seed + i + 300) * titles.length);

        results.push({
            title: titles[titleIdx],
            url: 'http://' + site.domain + urlPaths[pathIdx],
            snippet: snippet,
            size: Math.floor(seededRandom(seed + i + 400) * 80 + 5) + 'k',
            cached: Math.floor(seededRandom(seed + i + 500) * 30 + 1) + ' ' +
                   (seededRandom(seed + i + 600) > 0.5 ? 'Dec' : 'Nov') + ' 2003'
        });
    }

    return results;
}

// Format number with commas (classic style)
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

// Get random search time (classic Google showed this)
function getSearchTime() {
    return (Math.random() * 0.5 + 0.1).toFixed(2);
}

// Simple form validation
function validateSearch(form) {
    var query = form.q.value;
    if (query.trim() === '') {
        return false;
    }
    return true;
}

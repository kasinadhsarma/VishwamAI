import React, { useState, useEffect } from 'react';
import { Box, Heading, Text, Button, Input, Badge, Switch, Tabs, TabList, TabPanels, TabPanel, Tab, Menu, MenuButton, MenuList, MenuItem } from '@chakra-ui/react';
import { Leaf, Droplet, Wind, Flower, Globe, Search, Menu as MenuIcon, Moon, Sun, ArrowRight } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const Header = ({ darkMode, setDarkMode }) => {
  const [searchVisible, setSearchVisible] = useState(false);

  return (
    <Box
      mb={6}
      p={4}
      bg={darkMode ? 'gray.800' : 'linear-gradient(to-r, green.400, blue.500)'}
      rounded="lg"
      shadow="lg"
      color="white"
      transition="background-color 0.3s"
    >
      <Box display="flex" justifyContent="space-between" alignItems="center">
        <Heading as="h1" size="lg" fontWeight="bold">
          Oxygen Awareness Hub
        </Heading>
        <Box display="flex" alignItems="center" spacing={4}>
          <Button variant="ghost" size="sm" onClick={() => setSearchVisible(!searchVisible)}>
            <Search />
          </Button>
          <Badge variant="solid" display={{ base: 'none', md: 'inline-flex' }}>
            Air Quality: Good
          </Badge>
          <Switch
            isChecked={darkMode}
            onChange={setDarkMode}
            colorScheme="gray"
            icon={darkMode ? <Moon /> : <Sun />}
          />
          <Menu>
            <MenuButton as={Button} variant="ghost" size="sm" display={{ base: 'inline-flex', md: 'none' }}>
              <MenuIcon />
            </MenuButton>
            <MenuList>
              <MenuItem>Air Quality: Good</MenuItem>
              <MenuItem>Latest News</MenuItem>
              <MenuItem>Take Action</MenuItem>
            </MenuList>
          </Menu>
        </Box>
      </Box>
      {searchVisible && (
        <Box mt={4}>
          <Input placeholder="Search topics..." bg="whiteAlpha.200" borderColor="whiteAlpha.300" color="white" />
        </Box>
      )}
    </Box>
  );
};

const HeroSection = ({ darkMode }) => {
  return (
    <div className={`mb-8 p-8 ${darkMode ? 'bg-gray-800 text-white' : 'bg-blue-50'} rounded-lg shadow-lg transition-colors duration-300`}>
      <h2 className="text-3xl md:text-4xl font-bold mb-4 animate-fade-in">Breathe Easy, Act Wisely</h2>
      <p className="text-lg mb-6 animate-fade-in delay-200">Join us in our mission to raise awareness about the importance of oxygen and take action for a sustainable future.</p>
      <Button className="animate-fade-in delay-400">
        Get Involved <ArrowRight className="ml-2 h-4 w-4" />
      </Button>
    </div>
  );
};

// Replace static data with dynamic data from exampleData
const OxygenLevelsChart = ({ darkMode, data }) => {
  console.log('OxygenLevelsChart rendered');
  console.log('Received data:', data);

  return (
    <div className="h-64 mt-4">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <XAxis dataKey="year" stroke={darkMode ? "#fff" : "#000"} />
          <YAxis stroke={darkMode ? "#fff" : "#000"} />
          <Tooltip contentStyle={{ backgroundColor: darkMode ? '#333' : '#fff', color: darkMode ? '#fff' : '#000' }} />
          <Line type="monotone" dataKey="level" stroke="#82ca9d" strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

// Update AwarenessModule to pass exampleData to OxygenLevelsChart
const AwarenessModule = ({ darkMode, exampleData }) => {
  return (
    <Box className={`w-full overflow-hidden transition-all duration-300 hover:shadow-lg ${darkMode ? 'bg-gray-700 text-white' : ''}`}>
      <Box className={darkMode ? 'bg-gray-800' : 'bg-gradient-to-r from-green-400 to-blue-500 text-white'}>
        <Heading className="text-lg md:text-xl font-semibold">Oxygen Awareness</Heading>
        <Text className={darkMode ? 'text-gray-300' : 'text-white/80'}>Understand the importance of oxygen for our planet</Text>
      </Box>
      <Box className="p-6">
        <h3 className="text-lg font-semibold mb-4">Atmospheric Oxygen Levels</h3>
        <p className="mb-4">Oxygen levels in our atmosphere have been changing over time. Here's a look at the trends:</p>
        <OxygenLevelsChart darkMode={darkMode} data={exampleData} />
        <ul className="list-disc list-inside my-6 space-y-2">
          <li>Oxygen is crucial for most life forms on Earth</li>
          <li>It's produced by plants through photosynthesis</li>
          <li>Human activities are affecting oxygen levels</li>
          <li>Oceans produce about 50-80% of the Earth's oxygen</li>
        </ul>
        {exampleData && (
          <Box className="mt-6">
            <Heading className="text-lg font-semibold mb-4">Fetched Data from Backend</Heading>
            <pre className="bg-gray-100 p-4 rounded">{JSON.stringify(exampleData, null, 2)}</pre>
          </Box>
        )}
        <Button className={`w-full ${darkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-gradient-to-r from-green-400 to-blue-500 hover:from-green-500 hover:to-blue-600'} text-white transition-all duration-300 transform hover:scale-105`}>
          Learn More About Oxygen
        </Button>
      </Box>
    </Box>
  );
};

const PortfolioItem = ({ title, description, image, darkMode }) => (
  <Box className={`overflow-hidden transition-all duration-300 hover:shadow-lg ${darkMode ? 'bg-gray-700 text-white' : ''}`}>
    <img src={`/api/placeholder/${image}`} alt={title} className="w-full h-48 object-cover" />
    <Box className="p-4">
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <p className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>{description}</p>
    </Box>
    <Box>
      <Button variant="outline" className="w-full">Learn More</Button>
    </Box>
  </Box>
);

const PortfolioSection = ({ darkMode }) => (
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
    <PortfolioItem
      title="Reforestation Project"
      description="Planting trees to increase oxygen production and combat climate change."
      image="400/300"
      darkMode={darkMode}
    />
    <PortfolioItem
      title="Ocean Conservation"
      description="Protecting marine ecosystems that are vital for oxygen generation."
      image="400/300"
      darkMode={darkMode}
    />
    <PortfolioItem
      title="Clean Air Initiative"
      description="Working with communities to reduce air pollution and improve air quality."
      image="400/300"
      darkMode={darkMode}
    />
  </div>
);

const OxygenAwarenessDashboard = () => {
  const [darkMode, setDarkMode] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const [exampleData, setExampleData] = useState(null);

  useEffect(() => {
    // Simulating an initialization process
    const timer = setTimeout(() => {
      setIsInitialized(true);
    }, 2000);

    // Fetch data from the backend API
    fetch('http://10.240.250.104:8000/example/')
      .then(response => response.json())
      .then(data => {
        console.log('Fetched data:', data);
        setExampleData(data);
      })
      .catch(error => {
        console.error('Error fetching data:', error);
      });

    return () => clearTimeout(timer);
  }, []);

  if (!isInitialized) {
    return (
      <div className="flex items-center justify-center h-screen bg-gradient-to-r from-green-400 to-blue-500">
        <div className="text-white text-center">
          <h2 className="text-3xl font-bold mb-4">Initializing Oxygen Awareness Hub</h2>
          <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-white mx-auto"></div>
        </div>
      </div>
    );
  }

  return (
    <div className={`p-4 md:p-6 max-w-6xl mx-auto ${darkMode ? 'bg-gray-900' : 'bg-gray-50'} min-h-screen transition-colors duration-300`}>
      <Header darkMode={darkMode} setDarkMode={setDarkMode} />
      <HeroSection darkMode={darkMode} />

      <Tabs defaultValue="awareness" className="space-y-6">
        <TabList className={`flex md:inline-flex w-full ${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg p-1 shadow-sm overflow-x-auto md:overflow-x-visible`}>
          <Tab className={`flex-shrink-0 ${darkMode ? 'data-[state=active]:bg-blue-600' : 'data-[state=active]:bg-gradient-to-r data-[state=active]:from-green-400 data-[state=active]:to-blue-500'} data-[state=active]:text-white transition-all duration-300`}>
            <Leaf className="w-4 h-4 mr-2" />
            <span className="hidden md:inline">Awareness</span>
          </Tab>
          <Tab className={`flex-shrink-0 ${darkMode ? 'data-[state=active]:bg-blue-600' : 'data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-400 data-[state=active]:to-indigo-500'} data-[state=active]:text-white transition-all duration-300`}>
            <Droplet className="w-4 h-4 mr-2" />
            <span className="hidden md:inline">Water & Oxygen</span>
          </Tab>
          <Tab className={`flex-shrink-0 ${darkMode ? 'data-[state=active]:bg-blue-600' : 'data-[state=active]:bg-gradient-to-r data-[state=active]:from-yellow-400 data-[state=active]:to-orange-500'} data-[state=active]:text-white transition-all duration-300`}>
            <Wind className="w-4 h-4 mr-2" />
            <span className="hidden md:inline">Air Quality</span>
          </Tab>
          <Tab className={`flex-shrink-0 ${darkMode ? 'data-[state=active]:bg-blue-600' : 'data-[state=active]:bg-gradient-to-r data-[state=active]:from-green-400 data-[state=active]:to-teal-500'} data-[state=active]:text-white transition-all duration-300`}>
            <Flower className="w-4 h-4 mr-2" />
            <span className="hidden md:inline">Flora & O2</span>
          </Tab>
          <Tab className={`flex-shrink-0 ${darkMode ? 'data-[state=active]:bg-blue-600' : 'data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-400 data-[state=active]:to-pink-500'} data-[state=active]:text-white transition-all duration-300`}>
            <Globe className="w-4 h-4 mr-2" />
            <span className="hidden md:inline">Take Action</span>
          </Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            <Box className={`w-full overflow-hidden transition-all duration-300 hover:shadow-lg ${darkMode ? 'bg-gray-700 text-white' : ''}`}>
              <Box className={darkMode ? 'bg-gray-800' : 'bg-gradient-to-r from-green-400 to-blue-500 text-white'}>
                <Heading className="text-lg md:text-xl font-semibold">Oxygen Awareness</Heading>
                <Text className={darkMode ? 'text-gray-300' : 'text-white/80'}>Understand the importance of oxygen for our planet</Text>
              </Box>
              <Box className="p-6">
                <h3 className="text-lg font-semibold mb-4">Atmospheric Oxygen Levels</h3>
                <p className="mb-4">Oxygen levels in our atmosphere have been changing over time. Here's a look at the trends:</p>
                <OxygenLevelsChart darkMode={darkMode} data={exampleData} />
                <ul className="list-disc list-inside my-6 space-y-2">
                  <li>Oxygen is crucial for most life forms on Earth</li>
                  <li>It's produced by plants through photosynthesis</li>
                  <li>Human activities are affecting oxygen levels</li>
                  <li>Oceans produce about 50-80% of the Earth's oxygen</li>
                </ul>
                <Button className={`w-full ${darkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-gradient-to-r from-green-400 to-blue-500 hover:from-green-500 hover:to-blue-600'} text-white transition-all duration-300 transform hover:scale-105`}>
                  Learn More About Oxygen
                </Button>
              </Box>
            </Box>
          </TabPanel>

          <TabPanel>
            <Box className={`w-full overflow-hidden transition-all duration-300 hover:shadow-lg ${darkMode ? 'bg-gray-700 text-white' : ''}`}>
              <Box className={darkMode ? 'bg-gray-800' : 'bg-gradient-to-r from-purple-400 to-pink-500 text-white'}>
                <Heading className="text-lg md:text-xl font-semibold">Our Initiatives</Heading>
                <Text className={darkMode ? 'text-gray-300' : 'text-white/80'}>Explore our projects and get involved</Text>
              </Box>
              <Box className="p-6">
                <PortfolioSection darkMode={darkMode} />
              </Box>
            </Box>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </div>
  );
};

export default OxygenAwarenessDashboard;

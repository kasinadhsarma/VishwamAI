import React, { useState, useEffect } from 'react';
import { Box, Heading, Text, Button, Input, Badge, Switch, Tabs, TabList, TabPanels, TabPanel, Tab, Menu, MenuButton, MenuList, MenuItem } from '@chakra-ui/react';
import { Leaf, Droplet, Wind, Flower, Globe, Search, Menu as MenuIcon, Moon, Sun, ArrowRight } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const Header = ({ darkMode, setDarkMode }) => {
  const [searchVisible, setSearchVisible] = useState(false);

  return (
    <div className={`mb-6 p-4 ${darkMode ? 'bg-gray-800' : 'bg-gradient-to-r from-green-400 to-blue-500'} rounded-lg shadow-lg text-white transition-colors duration-300`}>
      <div className="flex justify-between items-center">
        <h1 className="text-xl md:text-2xl font-bold">Oxygen Awareness Hub</h1>
        <div className="flex items-center space-x-2 md:space-x-4">
          <Button variant="ghost" size="icon" onClick={() => setSearchVisible(!searchVisible)} className="text-white hover:bg-white/20">
            <Search className="h-5 w-5" />
          </Button>
          <Badge variant="secondary" className="hidden md:inline-flex">
            Air Quality: Good
          </Badge>
          <Switch
            checked={darkMode}
            onCheckedChange={setDarkMode}
            className="data-[state=checked]:bg-gray-600"
            icon={darkMode ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
          />
          <Menu>
            <MenuButton as={Button} variant="ghost" size="icon" className="md:hidden text-white hover:bg-white/20">
              <MenuIcon />
            </MenuButton>
            <MenuList align="end">
              <MenuItem>Air Quality: Good</MenuItem>
              <MenuItem>Latest News</MenuItem>
              <MenuItem>Take Action</MenuItem>
            </MenuList>
          </Menu>
        </div>
      </div>
      {searchVisible && (
        <div className="mt-4">
          <Input placeholder="Search topics..." className="bg-white/10 border-white/20 text-white placeholder-white/50" />
        </div>
      )}
    </div>
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

const OxygenLevelsChart = ({ darkMode }) => {
  const data = [
    { year: '1960', level: 315 },
    { year: '1980', level: 338 },
    { year: '2000', level: 369 },
    { year: '2020', level: 413 },
    { year: '2023', level: 420 },
  ];

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

const AwarenessModule = ({ darkMode }) => {
  return (
    <Box className={`w-full overflow-hidden transition-all duration-300 hover:shadow-lg ${darkMode ? 'bg-gray-700 text-white' : ''}`}>
      <Box className={darkMode ? 'bg-gray-800' : 'bg-gradient-to-r from-green-400 to-blue-500 text-white'}>
        <Heading className="text-lg md:text-xl font-semibold">Oxygen Awareness</Heading>
        <Text className={darkMode ? 'text-gray-300' : 'text-white/80'}>Understand the importance of oxygen for our planet</Text>
      </Box>
      <Box className="p-6">
        <h3 className="text-lg font-semibold mb-4">Atmospheric Oxygen Levels</h3>
        <p className="mb-4">Oxygen levels in our atmosphere have been changing over time. Here's a look at the trends:</p>
        <OxygenLevelsChart darkMode={darkMode} />
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

  useEffect(() => {
    // Simulating an initialization process
    const timer = setTimeout(() => {
      setIsInitialized(true);
    }, 2000);

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
                <OxygenLevelsChart darkMode={darkMode} />
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

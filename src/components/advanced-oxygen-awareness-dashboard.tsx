import React, { useState, useEffect } from 'react';
import { Box, Heading, Text, Button, Input, Badge, Switch, Tabs, TabList, TabPanels, Tab, TabPanel } from '@chakra-ui/react';
import { Leaf, Droplet, Wind, Flower, Globe, Search, Moon, Sun, ArrowRight } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const Header = ({ darkMode, setDarkMode }) => {
  const [searchVisible, setSearchVisible] = useState(false);

  return (
    <Box mb={6} p={4} bg={darkMode ? 'gray.800' : 'linear-gradient(to right, green.400, blue.500)'} rounded="lg" shadow="lg" color="white" transition="background-color 0.3s">
      <Box display="flex" justifyContent="space-between" alignItems="center">
        <Heading as="h1" size="xl">Oxygen Awareness Hub</Heading>
        <Box display="flex" alignItems="center" gap={4}>
          <Button variant="ghost" size="sm" onClick={() => setSearchVisible(!searchVisible)} colorScheme="whiteAlpha">
            <Search className="h-5 w-5" />
          </Button>
          <Badge variant="solid" colorScheme="green" display={{ base: 'none', md: 'inline-flex' }}>
            Air Quality: Good
          </Badge>
          <Switch isChecked={darkMode} onChange={setDarkMode} colorScheme="gray">
            {darkMode ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
          </Switch>
        </Box>
      </Box>
      {searchVisible && (
        <Box mt={4}>
          <Input placeholder="Search topics..." bg="whiteAlpha.200" borderColor="whiteAlpha.500" color="white" />
        </Box>
      )}
    </Box>
  );
};

const HeroSection = ({ darkMode }) => {
  return (
    <Box mb={8} p={8} bg={darkMode ? 'gray.800' : 'blue.50'} rounded="lg" shadow="lg" transition="background-color 0.3s">
      <Heading as="h2" size="2xl" mb={4}>Breathe Easy, Act Wisely</Heading>
      <Text fontSize="lg" mb={6}>Join us in our mission to raise awareness about the importance of oxygen and take action for a sustainable future.</Text>
      <Button rightIcon={<ArrowRight />} colorScheme="blue">
        Get Involved
      </Button>
    </Box>
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
    <Box h="64" mt={4}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <XAxis dataKey="year" stroke={darkMode ? "#fff" : "#000"} />
          <YAxis stroke={darkMode ? "#fff" : "#000"} />
          <Tooltip contentStyle={{ backgroundColor: darkMode ? '#333' : '#fff', color: darkMode ? '#fff' : '#000' }} />
          <Line type="monotone" dataKey="level" stroke="#82ca9d" strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
};

const AwarenessModule = ({ darkMode }) => {
  return (
    <Box w="full" overflow="hidden" transition="all 0.3s" _hover={{ shadow: 'lg' }} bg={darkMode ? 'gray.700' : 'white'} color={darkMode ? 'white' : 'black'}>
      <Box bg={darkMode ? 'gray.800' : 'linear-gradient(to right, green.400, blue.500)'} p={6}>
        <Heading as="h3" size="lg">Oxygen Awareness</Heading>
        <Text color={darkMode ? 'gray.300' : 'whiteAlpha.800'}>Understand the importance of oxygen for our planet</Text>
      </Box>
      <Box p={6}>
        <Heading as="h4" size="md" mb={4}>Atmospheric Oxygen Levels</Heading>
        <Text mb={4}>Oxygen levels in our atmosphere have been changing over time. Here's a look at the trends:</Text>
        <OxygenLevelsChart darkMode={darkMode} />
        <Box as="ul" listStyleType="disc" pl={4} my={6} gap={2}>
          <Box as="li">Oxygen is crucial for most life forms on Earth</Box>
          <Box as="li">It's produced by plants through photosynthesis</Box>
          <Box as="li">Human activities are affecting oxygen levels</Box>
          <Box as="li">Oceans produce about 50-80% of the Earth's oxygen</Box>
        </Box>
        <Button w="full" bgGradient="linear(to-r, green.400, blue.500)" _hover={{ bgGradient: 'linear(to-r, green.500, blue.600)' }} color="white" transition="all 0.3s" transform="scale(1.05)">
          Learn More About Oxygen
        </Button>
      </Box>
    </Box>
  );
};

const PortfolioItem = ({ title, description, image, darkMode }) => (
  <Box overflow="hidden" transition="all 0.3s" _hover={{ shadow: 'lg' }} bg={darkMode ? 'gray.700' : 'white'} color={darkMode ? 'white' : 'black'}>
    <img src={`/api/placeholder/${image}`} alt={title} className="w-full h-48 object-cover" />
    <Box p={4}>
      <Heading as="h3" size="md" mb={2}>{title}</Heading>
      <Text fontSize="sm" color={darkMode ? 'gray.300' : 'gray.600'}>{description}</Text>
    </Box>
    <Box p={4}>
      <Button variant="outline" w="full">Learn More</Button>
    </Box>
  </Box>
);

const PortfolioSection = ({ darkMode }) => (
  <Box display="grid" gridTemplateColumns={{ base: '1fr', md: 'repeat(2, 1fr)', lg: 'repeat(3, 1fr)' }} gap={6}>
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
  </Box>
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
      <Box display="flex" alignItems="center" justifyContent="center" h="100vh" bgGradient="linear(to-r, green.400, blue.500)">
        <Box textAlign="center" color="white">
          <Heading as="h2" size="xl" mb={4}>Initializing Oxygen Awareness Hub</Heading>
          <Box className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-white mx-auto"></Box>
        </Box>
      </Box>
    );
  }

  return (
    <Box p={4} maxW="6xl" mx="auto" bg={darkMode ? 'gray.900' : 'gray.50'} minH="100vh" transition="background-color 0.3s">
      <Header darkMode={darkMode} setDarkMode={setDarkMode} />
      <HeroSection darkMode={darkMode} />

      <Tabs variant="enclosed" isFitted>
        <TabList mb="1em">
          <Tab>
            <Leaf className="w-4 h-4 mr-2" />
            Awareness
          </Tab>
          <Tab>
            <Droplet className="w-4 h-4 mr-2" />
            Water & Oxygen
          </Tab>
          <Tab>
            <Wind className="w-4 h-4 mr-2" />
            Air Quality
          </Tab>
          <Tab>
            <Flower className="w-4 h-4 mr-2" />
            Flora & O2
          </Tab>
          <Tab>
            <Globe className="w-4 h-4 mr-2" />
            Take Action
          </Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            <AwarenessModule darkMode={darkMode} />
          </TabPanel>

          <TabPanel>
            <Box w="full" overflow="hidden" transition="all 0.3s" _hover={{ shadow: 'lg' }} bg={darkMode ? 'gray.700' : 'white'} color={darkMode ? 'white' : 'black'}>
              <Box bg={darkMode ? 'gray.800' : 'linear-gradient(to right, purple.400, pink.500)'} p={6}>
                <Heading as="h3" size="lg">Our Initiatives</Heading>
                <Text color={darkMode ? 'gray.300' : 'whiteAlpha.800'}>Explore our projects and get involved</Text>
              </Box>
              <Box p={6}>
                <PortfolioSection darkMode={darkMode} />
              </Box>
            </Box>
          </TabPanel>

          {/* Other tab contents would go here */}
        </TabPanels>
      </Tabs>
    </Box>
  );
};

export default OxygenAwarenessDashboard;

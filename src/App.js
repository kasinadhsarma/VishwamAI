import React, { useEffect } from 'react';
import { ChakraProvider, Box, Heading, VStack, Button, useColorMode, IconButton, Image } from '@chakra-ui/react';
import { BrowserRouter as Router, Route, Routes, Link as RouterLink } from 'react-router-dom';
import { FaSun, FaMoon } from 'react-icons/fa';
import logo from './assets/o2-awareness-logo.png';
import EducationalModule from './components/EducationalModule';
import SimulationModule from './components/SimulationModule';
import SustainabilityModule from './components/SustainabilityModule';
import FutureScenarioPlanningTool from './components/FutureScenarioPlanningTool';
import CommunityForum from './components/CommunityForum';
import AdvancedOxygenAwarenessDashboard from './AdvancedOxygenAwarenessDashboard.tsx';
import theme from './theme';

function App() {
  const { colorMode, toggleColorMode } = useColorMode();

  console.log('App component rendered');
  console.log('Initial color mode:', colorMode);
  console.log('toggleColorMode function:', toggleColorMode);

  const handleToggleColorMode = () => {
    console.log('Current color mode:', colorMode);
    if (typeof toggleColorMode === 'function') {
      toggleColorMode();
      console.log('Color mode after toggle:', colorMode === 'light' ? 'dark' : 'light');
    } else {
      console.error('toggleColorMode is not a function');
    }
  };

  useEffect(() => {
    console.log('Color mode updated:', colorMode);
  }, [colorMode]);

  return (
    <ChakraProvider theme={theme} key={colorMode}>
      <Router>
        <Box p={{ base: 4, md: 8 }} maxW="1200px" mx="auto">
          <Box as="header" mb={{ base: 4, md: 8 }} display="flex" justifyContent="space-between" alignItems="center">
            <Box display="flex" alignItems="center">
              <Image src={logo} alt="logo" width="100%" maxWidth="100px" mr={4} />
              <Heading as="h1" size={{ base: 'lg', md: 'xl' }}>
                Oxygen Agent
              </Heading>
            </Box>
            <IconButton
              aria-label="Toggle dark mode"
              icon={colorMode === 'light' ? <FaMoon /> : <FaSun />}
              onClick={handleToggleColorMode}
              mb={4}
            />
          </Box>
          <Box mb={{ base: 4, md: 8 }}>
            <Heading as="h2" size={{ base: 'md', md: 'lg' }} mb={4}>
              Breathe Easy, Act Wisely
            </Heading>
            <Button as={RouterLink} to="/community-forum" colorScheme="teal" size={{ base: 'md', md: 'lg' }}>
              Get Involved
            </Button>
          </Box>
          <Box>
            <Routes>
              <Route path="/" element={<EducationalModule />} />
              <Route path="/educational" element={<EducationalModule />} />
              <Route path="/simulation" element={<SimulationModule />} />
              <Route path="/sustainability" element={<SustainabilityModule />} />
              <Route path="/future-planning" element={<FutureScenarioPlanningTool />} />
              <Route path="/community-forum" element={<CommunityForum />} />
              <Route path="/dashboard" element={<AdvancedOxygenAwarenessDashboard />} />
            </Routes>
          </Box>
        </Box>
      </Router>
    </ChakraProvider>
  );
}

export default App;

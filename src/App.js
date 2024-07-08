import React from 'react';
import { ChakraProvider, Box, Heading, VStack, Button, useColorMode, IconButton, Image } from '@chakra-ui/react';
import { BrowserRouter as Router, Route, Routes, Link as RouterLink } from 'react-router-dom';
import { FaSun, FaMoon } from 'react-icons/fa';
import logo from './assets/o2-awareness-logo.png';
import EducationalModule from './components/EducationalModule';
import SimulationModule from './components/SimulationModule';
import SustainabilityModule from './components/SustainabilityModule';
import FutureScenarioPlanningTool from './components/FutureScenarioPlanningTool';
import CommunityForum from './components/CommunityForum';
import AdvancedOxygenAwarenessDashboard from './AdvancedOxygenAwarenessDashboard';
import theme from './theme';

function App() {
  const { colorMode, toggleColorMode } = useColorMode();

  return (
    <ChakraProvider theme={theme}>
      <Router>
        <Box p={{ base: 4, md: 8 }} maxW="1200px" mx="auto">
          <Box as="header" mb={{ base: 4, md: 8 }}>
            <Image src={logo} alt="logo" width="100%" maxWidth="200px" />
            <Heading as="h1" size={{ base: 'lg', md: 'xl' }} mb={4}>
              Oxygen Agent
            </Heading>
            <IconButton
              aria-label="Toggle dark mode"
              icon={colorMode === 'light' ? <FaMoon /> : <FaSun />}
              onClick={toggleColorMode}
              mb={4}
            />
            <VStack spacing={4} w="full">
              <Button as={RouterLink} to="/educational" colorScheme="teal" size={{ base: 'md', md: 'lg' }} w="full">
                Educational Module
              </Button>
              <Button as={RouterLink} to="/simulation" colorScheme="teal" size={{ base: 'md', md: 'lg' }} w="full">
                Simulation Module
              </Button>
              <Button as={RouterLink} to="/sustainability" colorScheme="teal" size={{ base: 'md', md: 'lg' }} w="full">
                Sustainability Module
              </Button>
              <Button as={RouterLink} to="/future-planning" colorScheme="teal" size={{ base: 'md', md: 'lg' }} w="full">
                Future Scenario Planning Tool
              </Button>
              <Button as={RouterLink} to="/community-forum" colorScheme="teal" size={{ base: 'md', md: 'lg' }} w="full">
                Community Forum
              </Button>
              <Button as={RouterLink} to="/dashboard" colorScheme="teal" size={{ base: 'md', md: 'lg' }} w="full">
                Advanced Oxygen Awareness Dashboard
              </Button>
            </VStack>
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
